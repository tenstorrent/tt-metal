//! The host half of Engram, bit-exact with `tt/engram_lookup.py` (`EngramHostLookup`) and the checkpoint's
//! `inference/engram.py` (`NgramHashState`): token ids -> compressed ids -> n-gram hashes -> fp8 table rows ->
//! bf16.
//!
//! Everything the checkpoint derives (the compressed-token map, the per-layer multipliers, primes and bucket
//! offsets) is handed in from Python's `NgramHashState`, so only the per-step arithmetic lives here.

use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use rayon::prelude::*;

/// A token that takes no part in an n-gram (an image-span token), as `NgramHashState.DEAD`.
pub const DEAD: i64 = -1;
/// One E8M0 scale per this many consecutive values of a table row.
pub const TABLE_BLOCK: usize = 32;

/// One Engram layer's table in host memory, as stored: `weight` `[rows, head_dim]` fp8 E4M3 and `scale`
/// `[rows, head_dim / 32]` E8M0. The memory belongs to Python (torch tensors it keeps alive).
#[derive(Clone, Copy)]
pub struct Table {
    pub weight: *const u8,
    pub scale: *const u8,
    pub rows: usize,
}
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

pub struct Engram {
    token_map: Vec<i64>,
    vocab_size: i64,
    pad_id: i64,
    max_ngram: usize,
    n_heads: usize,
    head_dim: usize,
    /// Every Engram layer of the checkpoint's layout, in layout order (hashes come out in this order).
    layout_layers: Vec<i64>,
    multipliers: Vec<i64>, // [layout layer][max_ngram]
    primes: Vec<i64>,      // [layout layer][n_cols]
    offsets: Vec<i64>,     // [layout layer][n_cols]
    /// (layout index, table) of each layer with a table, in table order.
    tables: Vec<(usize, Table)>,
    table_layers: Vec<i64>,
    batch: usize,
    max_seq: usize,
    histories: Vec<Mutex<Vec<i64>>>, // [slot] -> [batch * max_seq] compressed ids
    current: AtomicUsize,
}

/// fp8 E4M3FN -> f32 (no infinities; 0x7F/0xFF are NaN), in arithmetic and selects only so the dequantize loop
/// vectorizes (a table lookup does not).
#[inline(always)]
fn fp8(v: u8) -> f32 {
    let v = v as u32;
    let sign = (v & 0x80) << 24;
    let exp = (v >> 3) & 0xF;
    let mant = v & 7;
    let normal = sign | ((exp + 120) << 23) | (mant << 20);
    let subnormal = sign | (mant as f32 * (1.0 / 512.0)).to_bits();
    let bits = if v & 0x7F == 0x7F {
        0x7FC0_0000
    } else if exp == 0 {
        subnormal
    } else {
        normal
    };
    f32::from_bits(bits)
}

#[inline(always)]
fn e8m0(scale: u8) -> f32 {
    match scale {
        255 => f32::NAN,
        0 => f32::from_bits(0x0040_0000), // 2^-127, an f32 subnormal
        e => f32::from_bits((e as u32) << 23),
    }
}

/// f32 -> bf16 bits, rounding to nearest even, as `torch.Tensor.to(torch.bfloat16)` on the CPU (which turns every
/// NaN, whatever its sign or payload, into 0xFFFF).
#[inline(always)]
fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    let rounded = ((u.wrapping_add(((u >> 16) & 1) + 0x7FFF)) >> 16) as u16;
    if x.is_nan() {
        0xFFFF
    } else {
        rounded
    }
}

#[inline(always)]
fn prefetch(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr.cast());
    }
}

pub struct EngramArgs {
    pub token_map: Vec<i64>,
    pub vocab_size: i64,
    pub pad_id: i64,
    pub max_ngram: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub layout_layers: Vec<i64>,
    pub multipliers: Vec<Vec<i64>>,
    pub primes: Vec<Vec<i64>>,
    pub offsets: Vec<Vec<i64>>,
    pub tables: Vec<(i64, Table)>,
    pub batch: usize,
    pub max_seq: usize,
    pub sessions: usize,
}

impl Engram {
    pub fn new(args: EngramArgs) -> Result<Self, String> {
        let n_layers = args.layout_layers.len();
        let n_cols = (args.max_ngram - 1) * args.n_heads;
        if args.head_dim % TABLE_BLOCK != 0 {
            return Err(format!("head_dim {} is not a multiple of {TABLE_BLOCK}", args.head_dim));
        }
        for (name, rows, width) in [
            ("multipliers", &args.multipliers, args.max_ngram),
            ("primes", &args.primes, n_cols),
            ("offsets", &args.offsets, n_cols),
        ] {
            if rows.len() != n_layers || rows.iter().any(|r| r.len() != width) {
                return Err(format!("{name} must be [{n_layers}][{width}]"));
            }
        }
        let mut tables = Vec::new();
        for (layer_id, table) in &args.tables {
            let k = args
                .layout_layers
                .iter()
                .position(|l| l == layer_id)
                .ok_or_else(|| format!("layer {layer_id} is not an Engram layer of the layout"))?;
            tables.push((k, *table));
        }
        let sessions = args.sessions.max(1);
        Ok(Self {
            token_map: args.token_map,
            vocab_size: args.vocab_size,
            pad_id: args.pad_id,
            max_ngram: args.max_ngram,
            n_heads: args.n_heads,
            head_dim: args.head_dim,
            layout_layers: args.layout_layers,
            multipliers: args.multipliers.concat(),
            primes: args.primes.concat(),
            offsets: args.offsets.concat(),
            table_layers: args.tables.iter().map(|(l, _)| *l).collect(),
            tables,
            batch: args.batch,
            max_seq: args.max_seq,
            histories: (0..sessions).map(|_| Mutex::new(vec![0i64; args.batch * args.max_seq])).collect(),
            current: AtomicUsize::new(0),
        })
    }

    pub fn n_cols(&self) -> usize {
        (self.max_ngram - 1) * self.n_heads
    }

    pub fn n_layout_layers(&self) -> usize {
        self.layout_layers.len()
    }

    /// Values of one layer's rows for one token: `n_cols * head_dim` (the device's `kin`).
    pub fn kin(&self) -> usize {
        self.n_cols() * self.head_dim
    }

    pub fn table_layers(&self) -> &[i64] {
        &self.table_layers
    }

    pub fn vocab_size(&self) -> i64 {
        self.vocab_size
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn use_session(&self, slot: usize) -> Result<(), String> {
        self.check_slot(slot)?;
        self.current.store(slot, Ordering::Release);
        Ok(())
    }

    pub fn current_session(&self) -> usize {
        self.current.load(Ordering::Acquire)
    }

    fn check_slot(&self, slot: usize) -> Result<(), String> {
        if slot >= self.histories.len() {
            return Err(format!("session slot {slot} is past the {} sessions", self.histories.len()));
        }
        Ok(())
    }

    pub fn export_history(&self, slot: usize, pos: usize) -> Result<Vec<i64>, String> {
        self.check_slot(slot)?;
        if pos > self.max_seq {
            return Err(format!("history position {pos} is past max_seq {}", self.max_seq));
        }
        let history = self.histories[slot].lock();
        Ok((0..self.batch).flat_map(|b| history[b * self.max_seq..b * self.max_seq + pos].iter().copied()).collect())
    }

    /// Make `values` (`[batch, len]`, an export of any slot) the start of slot `slot`'s history.
    pub fn import_history(&self, slot: usize, values: &[i64], len: usize) -> Result<(), String> {
        self.check_slot(slot)?;
        if values.len() != self.batch * len || len > self.max_seq {
            return Err(format!("history of {} values is not [{}, {len}] within max_seq", values.len(), self.batch));
        }
        let mut history = self.histories[slot].lock();
        for b in 0..self.batch {
            history[b * self.max_seq..b * self.max_seq + len].copy_from_slice(&values[b * len..(b + 1) * len]);
        }
        Ok(())
    }

    /// `NgramHashState.forward` against slot `slot`'s history: `ids` `[batch, len]` at positions `start ..`
    /// (ids past the vocabulary cached as dead) -> hashes `[batch, len, n_layout_layers, n_cols]`.
    pub fn hash(&self, slot: usize, ids: &[i64], start: usize) -> Result<Vec<i64>, String> {
        self.check_slot(slot)?;
        let mut history = self.histories[slot].lock();
        self.hash_history(&mut history, ids, start)
    }

    /// `hash` against a caller-owned history `[batch, max_seq]`, which it extends. A prefetch uses a copy so the
    /// lookup does not write a live session the decode packet writer is hashing.
    fn hash_history(&self, history: &mut [i64], ids: &[i64], start: usize) -> Result<Vec<i64>, String> {
        let batch = self.batch;
        if history.len() != batch * self.max_seq {
            return Err(format!(
                "history has {} values, not [batch={batch}, max_seq={}]",
                history.len(),
                self.max_seq
            ));
        }
        if ids.is_empty() || ids.len() % batch != 0 {
            return Err(format!("{} ids are not [batch={batch}, len]", ids.len()));
        }
        let len = ids.len() / batch;
        if start + len > self.max_seq {
            return Err(format!("positions {start}..{} are past max_seq {}", start + len, self.max_seq));
        }
        let (n_layers, n_cols, max_ngram, n_heads) =
            (self.layout_layers.len(), self.n_cols(), self.max_ngram, self.n_heads);
        for b in 0..batch {
            for l in 0..len {
                let id = ids[b * len + l];
                let compressed = if id >= self.vocab_size {
                    DEAD
                } else {
                    *self.token_map.get(id as usize).filter(|_| id >= 0).ok_or_else(|| {
                        format!("token id {id} is outside the {}-token compressed map", self.token_map.len())
                    })?
                };
                history[b * self.max_seq + start + l] = compressed;
            }
        }
        let mut out = vec![0i64; batch * len * n_layers * n_cols];
        let mut tokens = vec![0i64; max_ngram];
        for b in 0..batch {
            let row = &history[b * self.max_seq..(b + 1) * self.max_seq];
            for l in 0..len {
                let pos = start + l;
                let mut blocked = false;
                for (shift, token) in tokens.iter_mut().enumerate() {
                    let source = row[pos.saturating_sub(shift)];
                    blocked |= pos < shift || source == DEAD;
                    *token = if blocked { self.pad_id } else { source };
                }
                let base = (b * len + l) * n_layers * n_cols;
                for k in 0..n_layers {
                    let mult = &self.multipliers[k * max_ngram..(k + 1) * max_ngram];
                    let mut rolling = tokens[0].wrapping_mul(mult[0]);
                    for i in 1..max_ngram {
                        rolling ^= tokens[i].wrapping_mul(mult[i]);
                        for h in 0..n_heads {
                            let c = (i - 1) * n_heads + h;
                            out[base + k * n_cols + c] =
                                rolling.rem_euclid(self.primes[k * n_cols + c]) + self.offsets[k * n_cols + c];
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Dequantized rows `ids` of table `t` (an index into the tables, in table order), into `out`
    /// (`ids.len() * head_dim` bf16 bits).
    pub fn gather(&self, t: usize, ids: &[i64], out: &mut [u16]) -> Result<(), String> {
        let (_, table) = self.tables[t];
        let hd = self.head_dim;
        let blocks = hd / TABLE_BLOCK;
        debug_assert_eq!(out.len(), ids.len() * hd);
        for &id in ids {
            if id < 0 || id as usize >= table.rows {
                return Err(format!("row {id} is outside layer {}'s {}-row table", self.table_layers[t], table.rows));
            }
            let row = id as usize;
            unsafe {
                let w = table.weight.add(row * hd);
                for line in (0..hd).step_by(64) {
                    prefetch(w.add(line));
                }
                prefetch(table.scale.add(row * blocks));
            }
        }
        for (j, &id) in ids.iter().enumerate() {
            let row = id as usize;
            let weight = unsafe { std::slice::from_raw_parts(table.weight.add(row * hd), hd) };
            let scale = unsafe { std::slice::from_raw_parts(table.scale.add(row * blocks), blocks) };
            let dst = &mut out[j * hd..(j + 1) * hd];
            for (blk, (&s, w)) in scale.iter().zip(weight.chunks_exact(TABLE_BLOCK)).enumerate() {
                let s = e8m0(s);
                for (d, &v) in dst[blk * TABLE_BLOCK..(blk + 1) * TABLE_BLOCK].iter_mut().zip(w) {
                    *d = bf16(fp8(v) * s);
                }
            }
        }
        Ok(())
    }

    /// One token's rows of table `t`, `kin` bf16 values, from its hashes (`[n_layout_layers, n_cols]`);
    /// zeros for an image-span token.
    pub fn token_rows(&self, t: usize, id: i64, hashes: &[i64], out: &mut [u16]) -> Result<(), String> {
        if id >= self.vocab_size {
            out.fill(0);
            return Ok(());
        }
        let (k, _) = self.tables[t];
        let n_cols = self.n_cols();
        self.gather(t, &hashes[k * n_cols..(k + 1) * n_cols], out)
    }

    /// `EngramHostLookup.__call__`: hash `ids` `[batch, len]` at `start` against the current session (extending
    /// its history), then each table's rows `[batch, len, kin]` bf16, the positions spread over threads.
    /// `outs` holds one `ids.len() * kin` buffer per table, in table order.
    pub fn rows_into(&self, ids: &[i64], start: usize, outs: &mut [&mut [u16]]) -> Result<(), String> {
        let hashes = self.hash(self.current_session(), ids, start)?;
        self.write_token_rows(ids, &hashes, outs)
    }

    /// `rows_into` against a copy of `history` (`[batch, hist_len]`, positions `0 .. hist_len`), leaving every live
    /// session untouched. The server prefetches a waiting prompt's rows this way.
    pub fn rows_from_history(
        &self,
        history: &[i64],
        hist_len: usize,
        ids: &[i64],
        start: usize,
        outs: &mut [&mut [u16]],
    ) -> Result<(), String> {
        let batch = self.batch;
        if hist_len > self.max_seq || history.len() != batch * hist_len {
            return Err(format!(
                "history of {} values is not [batch={batch}, {hist_len}] within max_seq {}",
                history.len(),
                self.max_seq
            ));
        }
        let mut local = vec![0i64; batch * self.max_seq];
        for b in 0..batch {
            local[b * self.max_seq..b * self.max_seq + hist_len]
                .copy_from_slice(&history[b * hist_len..(b + 1) * hist_len]);
        }
        let hashes = self.hash_history(&mut local, ids, start)?;
        self.write_token_rows(ids, &hashes, outs)
    }

    fn write_token_rows(&self, ids: &[i64], hashes: &[i64], outs: &mut [&mut [u16]]) -> Result<(), String> {
        let kin = self.kin();
        if outs.len() != self.tables.len() || outs.iter().any(|o| o.len() != ids.len() * kin) {
            return Err(format!("need {} output buffers of {} values", self.tables.len(), ids.len() * kin));
        }
        let per_token = self.n_layout_layers() * self.n_cols();
        for (t, out) in outs.iter_mut().enumerate() {
            out.par_chunks_mut(kin).enumerate().try_for_each(|(n, dst)| {
                self.token_rows(t, ids[n], &hashes[n * per_token..(n + 1) * per_token], dst)
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_decodes() {
        assert_eq!(fp8(0x38), 1.0); // exp 7, mant 0
        assert_eq!(fp8(0xB8), -1.0);
        assert_eq!(fp8(0x7E), 448.0); // the largest finite E4M3FN
        assert_eq!(fp8(0x01), 2f32.powi(-9)); // the smallest subnormal
        assert!(fp8(0x7F).is_nan());
        assert!(fp8(0xFF).is_nan());
        assert_eq!(fp8(0x07), 7.0 * 2f32.powi(-9)); // the largest subnormal
        assert_eq!(fp8(0x08), 2f32.powi(-6)); // the smallest normal
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(bf16(1.0), 0x3F80);
        assert_eq!(bf16(f32::from_bits(0x3F80_8000)), 0x3F80); // tie, even stays
        assert_eq!(bf16(f32::from_bits(0x3F81_8000)), 0x3F82); // tie, odd rounds up
        assert_eq!(bf16(f32::from_bits(0x3F80_8001)), 0x3F81);
        assert_eq!(bf16(f32::from_bits(0x7FF0_0000)), 0xFFFF);
    }
}
