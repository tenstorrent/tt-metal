//! The traced decode step's host side on native threads: trace replay per pipeline stage, the step packet, the
//! Engram rows and the output, each on a thread of its own and none of them holding the GIL.
//!
//! * One dispatch thread per stage owns that submesh's command queue 0 and replays each posted step's trace
//!   there, so the stages are enqueued in parallel; every stage sees the steps in posting order, which is what
//!   pairs up the stage-to-stage sockets inside the traces.
//! * The packet thread writes each step's INT32 packet page into stage 0's H2D socket, then hashes the step's
//!   tokens (extending the session's Engram history) and hands the hashes to the Engram threads.
//! * One Engram thread per table layer gathers and dequantizes the step's rows and writes them, rolled per TP
//!   rank, into that layer's per-device H2D sockets.
//! * The output thread polls the D2H socket and queues each step's output page(s) as soon as they land.
//!
//! The sockets and submeshes are Python's (by address); while this runtime runs, Python must not write the
//! packet and Engram sockets or read the output socket, except with nothing in flight here (see `wait_idle`).
//! It queues work of its own on a stage only once `drain` says the stage threads have issued every replay
//! (speculative decode's ring fixups); a DSpark drafter's row is no stage, and its queue stays Python's.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, Sender};
use parking_lot::Mutex;

use crate::counters::{unix_ns, CounterMap};
use crate::engram::Engram;
use crate::ffi::{MeshDevice, D2H, H2D};

/// How long the output thread spins on an empty socket before it starts sleeping between polls.
const SPIN_BEFORE_SLEEP: Duration = Duration::from_millis(2);
const POLL_SLEEP: Duration = Duration::from_micros(20);

pub struct EngramSockets {
    /// Index of the layer's table in the [`Engram`] (table order).
    pub table: usize,
    /// One H2D socket per device of the layer's stage, in TP-rank order.
    pub ranks: Vec<H2D>,
    /// `kin / tp`: rank `r` gets each row rolled left by `r * local_kin` (`torch.roll(rows, -r * local_kin)`).
    pub local_kin: usize,
}

pub struct DecodeConfig {
    pub stages: Vec<MeshDevice>,
    /// `[variant][stage]` trace ids.
    pub variants: Vec<Vec<u32>>,
    pub packet: H2D,
    pub batch: usize,
    pub sliding_window: i64,
    pub win_slot_rows: i64,
    /// The DSpark drafter's ring depth per session slot: a step's row goes to ring row
    /// `slot * drafter_ring_rows + pos % drafter_ring_rows`; 0: no drafter.
    pub drafter_ring_rows: i64,
    pub engram: Option<Arc<Engram>>,
    pub engram_sockets: Vec<EngramSockets>,
    pub output: D2H,
    pub output_step_bytes: usize,
    /// CPUs to pin the threads to, in creation order (output, packet, Engram layers, stages); empty: no pinning.
    pub cpus: Vec<usize>,
    /// File the counters are kept in, for a monitor process to map (see [`crate::counters`]); `None`: private.
    pub counters_path: Option<String>,
}

struct Step {
    tokens: Vec<i64>,
    pos: i64,
    slot: i64,
}

struct Hashed {
    tokens: Vec<i64>,
    hashes: Vec<i64>,
}

struct Shared {
    error: Mutex<Option<String>>,
    failed: AtomicBool,
    stop: AtomicBool,
    issued: Vec<AtomicU64>,
    engram_written: Vec<AtomicU64>,
    c: CounterMap,
}

impl Shared {
    fn fail(&self, what: &str, message: String) {
        let mut error = self.error.lock();
        if error.is_none() {
            *error = Some(format!("{what}: {message}"));
        }
        self.failed.store(true, Ordering::Release);
    }

    fn check(&self) -> Result<(), String> {
        if self.failed.load(Ordering::Acquire) {
            return Err(self.error.lock().clone().unwrap_or_else(|| "decode host failed".into()));
        }
        Ok(())
    }
}

pub struct DecodeHost {
    shared: Arc<Shared>,
    variants: Vec<Vec<u32>>,
    stage_tx: Vec<Sender<Option<u32>>>,
    packet_tx: Sender<Option<Step>>,
    output_rx: Receiver<Vec<u8>>,
    batch: usize,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

fn pin(cpus: &[usize], index: usize) {
    if cpus.is_empty() {
        return;
    }
    let cpu = cpus[index % cpus.len()];
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

fn as_bytes(values: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() * 2) }
}

/// V4.1's packet fields after the tokens (`DeepSeekV41Model._build_packet`): the session slot's ring row, pos,
/// slot, ring base, window base and the drafter ring row, in the slot's rows of the drafter ring (no drafter: 0).
fn packet_fields(pos: i64, slot: i64, window: i64, win_rows: i64, drafter_ring_rows: i64) -> [i64; 6] {
    let drafter_row = if drafter_ring_rows > 0 { slot * drafter_ring_rows + pos % drafter_ring_rows } else { 0 };
    [slot * window + pos % window, pos, slot, slot * window, slot * win_rows, drafter_row]
}

impl DecodeHost {
    pub fn start(config: DecodeConfig) -> Result<Self, String> {
        let n_stages = config.stages.len();
        if config.variants.iter().any(|tids| tids.len() != n_stages) {
            return Err(format!("every variant needs one trace id per stage ({n_stages})"));
        }
        if config.output_step_bytes == 0 || config.output_step_bytes % config.output.page_size() != 0 {
            return Err(format!(
                "a step's {} B output is not whole {} B pages",
                config.output_step_bytes,
                config.output.page_size()
            ));
        }
        let engram = config.engram.clone();
        if !config.engram_sockets.is_empty() && engram.is_none() {
            return Err("Engram sockets given without an Engram lookup".into());
        }
        if let Some(engram) = &engram {
            if engram.batch() != config.batch {
                return Err(format!("the Engram lookup is batch {}, decode batch {}", engram.batch(), config.batch));
            }
            for layer in &config.engram_sockets {
                let page = engram.kin() * 2;
                if layer.ranks.iter().any(|s| s.page_size() != page) {
                    return Err(format!("an Engram socket's page is not one {page} B row"));
                }
            }
        }
        let pkt_ints = config.packet.page_size() / 4;
        if pkt_ints < config.batch + 6 {
            return Err(format!("a {pkt_ints}-word packet cannot hold {} tokens and the slot fields", config.batch));
        }

        let shared = Arc::new(Shared {
            error: Mutex::new(None),
            failed: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            issued: (0..n_stages).map(|_| AtomicU64::new(0)).collect(),
            engram_written: config.engram_sockets.iter().map(|_| AtomicU64::new(0)).collect(),
            c: CounterMap::open(config.counters_path.as_deref())?,
        });
        let cpus = config.cpus.clone();
        let mut threads = Vec::new();
        let mut spawn = |name: String, body: Box<dyn FnOnce() + Send>| -> Result<(), String> {
            let index = threads.len();
            let cpus = cpus.clone();
            let handle = std::thread::Builder::new()
                .name(name)
                .spawn(move || {
                    pin(&cpus, index);
                    body()
                })
                .map_err(|e| e.to_string())?;
            threads.push(handle);
            Ok(())
        };

        // Output reader.
        let (output_tx, output_rx) = unbounded::<Vec<u8>>();
        {
            let shared = shared.clone();
            let output = config.output;
            let step_bytes = config.output_step_bytes;
            let page = output.page_size();
            let pages = step_bytes / page;
            spawn(
                "dsv41-output".into(),
                Box::new(move || {
                    let mut idle_since: Option<Instant> = None;
                    // A step's pages read so far: a step can be more pages than the FIFO holds (logits without
                    // device sampling), so its pages are read as they come rather than once they are all in.
                    let mut out = vec![0u8; step_bytes];
                    let mut done = 0;
                    let mut t0 = Instant::now();
                    while !shared.stop.load(Ordering::Acquire) {
                        match output.pages_available() {
                            Err(e) => return shared.fail("output reader", e),
                            Ok(n) if n > 0 => {
                                if done == 0 {
                                    t0 = Instant::now();
                                }
                                let k = n.min(pages - done);
                                if let Err(e) = output.read(&mut out[done * page..(done + k) * page]) {
                                    return shared.fail("output reader", e);
                                }
                                done += k;
                                idle_since = None;
                                if done < pages {
                                    continue;
                                }
                                done = 0;
                                let out = std::mem::replace(&mut out, vec![0u8; step_bytes]);
                                shared.c.output_read.add(t0);
                                shared.c.last_output_ns.store(unix_ns(), Ordering::Relaxed);
                                shared.c.outputs_read.fetch_add(1, Ordering::AcqRel);
                                if output_tx.send(out).is_err() {
                                    return;
                                }
                            }
                            Ok(_) => {
                                let since = *idle_since.get_or_insert_with(Instant::now);
                                if since.elapsed() < SPIN_BEFORE_SLEEP {
                                    std::hint::spin_loop();
                                } else {
                                    std::thread::sleep(POLL_SLEEP);
                                }
                            }
                        }
                    }
                }),
            )?;
        }

        // Engram writers, fed by the packet thread.
        let mut engram_txs = Vec::new();
        for (i, layer) in config.engram_sockets.into_iter().enumerate() {
            let (tx, rx) = unbounded::<Option<Arc<Hashed>>>();
            engram_txs.push(tx);
            let shared = shared.clone();
            let engram = engram.clone().expect("checked above");
            let batch = config.batch;
            spawn(
                format!("dsv41-engram-{}", engram.table_layers()[layer.table]),
                Box::new(move || {
                    let kin = engram.kin();
                    let per_token = engram.n_layout_layers() * engram.n_cols();
                    let mut rows = vec![0u16; batch * kin];
                    let mut rolled = vec![0u16; batch * kin];
                    while let Ok(Some(step)) = rx.recv() {
                        let t0 = Instant::now();
                        for b in 0..batch {
                            let hashes = &step.hashes[b * per_token..(b + 1) * per_token];
                            if let Err(e) =
                                engram.token_rows(layer.table, step.tokens[b], hashes, &mut rows[b * kin..(b + 1) * kin])
                            {
                                return shared.fail("Engram rows", e);
                            }
                        }
                        for (rank, socket) in layer.ranks.iter().enumerate() {
                            let shift = (rank * layer.local_kin) % kin;
                            let data = if shift == 0 {
                                &rows
                            } else {
                                for b in 0..batch {
                                    let (src, dst) = (&rows[b * kin..(b + 1) * kin], &mut rolled[b * kin..(b + 1) * kin]);
                                    dst[..kin - shift].copy_from_slice(&src[shift..]);
                                    dst[kin - shift..].copy_from_slice(&src[..shift]);
                                }
                                &rolled
                            };
                            if let Err(e) = socket.write(as_bytes(data)) {
                                return shared.fail("Engram socket write", e);
                            }
                        }
                        shared.c.engram.add(t0);
                        shared.engram_written[i].fetch_add(1, Ordering::AcqRel);
                    }
                }),
            )?;
        }

        // Packet writer.
        let (packet_tx, packet_rx) = unbounded::<Option<Step>>();
        {
            let shared = shared.clone();
            let packet = config.packet;
            let (w, win_rows, drafter_rows) = (config.sliding_window, config.win_slot_rows, config.drafter_ring_rows);
            let engram = if engram_txs.is_empty() { None } else { engram.clone() };
            spawn(
                "dsv41-packet".into(),
                Box::new(move || {
                    let mut page = vec![0i32; pkt_ints];
                    while let Ok(Some(step)) = packet_rx.recv() {
                        let t0 = Instant::now();
                        page.fill(0);
                        let (pos, slot) = (step.pos, step.slot);
                        let fields = packet_fields(pos, slot, w, win_rows, drafter_rows);
                        for (dst, v) in page.iter_mut().zip(step.tokens.iter().chain(fields.iter())) {
                            *dst = *v as i32;
                        }
                        let bytes = unsafe { std::slice::from_raw_parts(page.as_ptr().cast::<u8>(), pkt_ints * 4) };
                        if let Err(e) = packet.write(bytes) {
                            return shared.fail("packet socket write", e);
                        }
                        shared.c.packet.add(t0);
                        shared.c.packets_written.fetch_add(1, Ordering::AcqRel);
                        if let Some(engram) = &engram {
                            let t1 = Instant::now();
                            let hashes = match engram.hash(slot as usize, &step.tokens, pos as usize) {
                                Ok(h) => h,
                                Err(e) => return shared.fail("Engram hash", e),
                            };
                            shared.c.hash.add(t1);
                            let hashed = Arc::new(Hashed { tokens: step.tokens, hashes });
                            for tx in &engram_txs {
                                let _ = tx.send(Some(hashed.clone()));
                            }
                        }
                    }
                    for tx in &engram_txs {
                        let _ = tx.send(None);
                    }
                }),
            )?;
        }

        // Stage dispatchers.
        let mut stage_tx = Vec::new();
        for (k, mesh) in config.stages.into_iter().enumerate() {
            let (tx, rx) = unbounded::<Option<u32>>();
            stage_tx.push(tx);
            let shared = shared.clone();
            spawn(
                format!("dsv41-stage-{k}"),
                Box::new(move || {
                    while let Ok(Some(trace_id)) = rx.recv() {
                        let t0 = Instant::now();
                        if let Err(e) = mesh.execute_trace(trace_id) {
                            return shared.fail(&format!("stage {k} trace replay"), e);
                        }
                        shared.c.dispatch.add(t0);
                        shared.issued[k].fetch_add(1, Ordering::AcqRel);
                    }
                }),
            )?;
        }

        Ok(Self {
            shared,
            variants: config.variants,
            stage_tx,
            packet_tx,
            output_rx,
            batch: config.batch,
            threads: Mutex::new(threads),
        })
    }

    /// Queue variant `variant`'s trace on every stage. Returns at once.
    pub fn dispatch(&self, variant: usize) -> Result<(), String> {
        self.shared.check()?;
        let tids = self.variants.get(variant).ok_or_else(|| format!("no trace variant {variant}"))?;
        self.shared.c.posted.fetch_add(1, Ordering::AcqRel);
        for (tx, &tid) in self.stage_tx.iter().zip(tids) {
            tx.send(Some(tid)).map_err(|_| "a stage dispatcher has stopped".to_string())?;
        }
        Ok(())
    }

    /// Queue one step's packet (and Engram rows): `tokens` (`batch` of them) at `pos` in session slot `slot`.
    pub fn write_step(&self, tokens: Vec<i64>, pos: i64, slot: i64) -> Result<(), String> {
        self.shared.check()?;
        if tokens.len() != self.batch {
            return Err(format!("expected {} tokens, got {}", self.batch, tokens.len()));
        }
        self.shared.c.queued.fetch_add(1, Ordering::AcqRel);
        self.packet_tx.send(Some(Step { tokens, pos, slot })).map_err(|_| "the packet writer has stopped".to_string())
    }

    fn wait_until(&self, done: impl Fn(&Shared) -> bool) -> Result<(), String> {
        let t0 = Instant::now();
        loop {
            self.shared.check()?;
            if done(&self.shared) {
                return Ok(());
            }
            if t0.elapsed() < SPIN_BEFORE_SLEEP {
                std::hint::spin_loop();
            } else {
                std::thread::sleep(POLL_SLEEP);
            }
        }
    }

    /// Wait until every queued trace replay has been issued (enqueued on its stage), not run.
    pub fn drain(&self) -> Result<(), String> {
        self.wait_until(|s| {
            let posted = s.c.posted.load(Ordering::Acquire);
            s.issued.iter().all(|i| i.load(Ordering::Acquire) == posted)
        })
    }

    /// Wait until every queued packet and its Engram rows are in their sockets.
    pub fn wait_written(&self) -> Result<(), String> {
        self.wait_until(|s| {
            let queued = s.c.queued.load(Ordering::Acquire);
            s.c.packets_written.load(Ordering::Acquire) == queued
                && s.engram_written.iter().all(|w| w.load(Ordering::Acquire) == queued)
        })
    }

    /// The oldest unread step output; blocks until one is in.
    pub fn read_output(&self) -> Result<Vec<u8>, String> {
        loop {
            match self.output_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(out) => return Ok(out),
                Err(RecvTimeoutError::Timeout) => self.shared.check()?,
                Err(RecvTimeoutError::Disconnected) => {
                    self.shared.check()?;
                    return Err("the output reader has stopped".into());
                }
            }
        }
    }

    pub fn outputs_ready(&self) -> usize {
        self.output_rx.len()
    }

    pub fn stats(&self) -> Vec<(&'static str, u64, f64)> {
        let c = &self.shared.c;
        let mut out = vec![
            ("posted", c.posted.load(Ordering::Relaxed), 0.0),
            ("queued", c.queued.load(Ordering::Relaxed), 0.0),
            ("outputs_read", c.outputs_read.load(Ordering::Relaxed), 0.0),
        ];
        for (name, timer) in [
            ("dispatch_us", &c.dispatch),
            ("packet_us", &c.packet),
            ("hash_us", &c.hash),
            ("engram_us", &c.engram),
            ("output_read_us", &c.output_read),
        ] {
            let (count, mean) = timer.mean_us();
            out.push((name, count, mean));
        }
        out
    }

    /// Stop every thread. Threads blocked in a socket write (a step that will never be consumed) are left
    /// detached.
    pub fn close(&self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.c.closed.store(1, Ordering::Release);
        let _ = self.packet_tx.send(None);
        for tx in &self.stage_tx {
            let _ = tx.send(None);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        for handle in self.threads.lock().drain(..) {
            while !handle.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            if handle.is_finished() {
                let _ = handle.join();
            }
        }
    }
}

impl Drop for DecodeHost {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::packet_fields;

    #[test]
    fn packet_fields_match_build_packet() {
        // Slot 2 of 128-row rings and 32-row windows, at pos 300.
        assert_eq!(packet_fields(300, 2, 128, 32, 0), [2 * 128 + 300 % 128, 300, 2, 256, 64, 0]);
        // With a 160-row drafter ring.
        assert_eq!(packet_fields(300, 0, 128, 32, 160), [300 % 128, 300, 0, 0, 0, 140]);
        assert_eq!(packet_fields(159, 0, 128, 32, 160)[5], 159);
        assert_eq!(packet_fields(160, 0, 128, 32, 160)[5], 0);
        // Each session slot has its own ring_rows rows of the drafter ring.
        assert_eq!(packet_fields(300, 2, 128, 32, 160)[5], 2 * 160 + 140);
    }
}
