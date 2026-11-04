//! The decode host's counters, kept in a shared mapping so a monitor in another process (`demo/monitor.py`) can
//! read them while the decode threads update them, with nothing asked of the server. The layout is fixed
//! (`#[repr(C)]`, every field a `u64`) and mirrored by `monitor.py`'s `COUNTER_FIELDS`; bump [`VERSION`] when
//! it changes.

use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// `b"DSV41MON"` read as a little-endian `u64`; written last, once every other header field is in.
pub const MAGIC: u64 = u64::from_le_bytes(*b"DSV41MON");
pub const VERSION: u64 = 1;

pub fn unix_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

#[repr(C)]
#[derive(Default)]
pub struct Timer {
    pub count: AtomicU64,
    pub nanos: AtomicU64,
}

impl Timer {
    pub fn add(&self, since: Instant) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.nanos.fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    pub fn mean_us(&self) -> (u64, f64) {
        let count = self.count.load(Ordering::Relaxed);
        let nanos = self.nanos.load(Ordering::Relaxed);
        (count, if count == 0 { 0.0 } else { nanos as f64 / count as f64 / 1e3 })
    }
}

#[repr(C)]
pub struct Counters {
    pub magic: AtomicU64,
    pub version: AtomicU64,
    pub pid: AtomicU64,
    pub started_ns: AtomicU64,
    /// Steps whose traces were queued (`DecodeHost::dispatch`).
    pub posted: AtomicU64,
    /// Steps whose packets were queued (`DecodeHost::write_step`).
    pub queued: AtomicU64,
    pub packets_written: AtomicU64,
    pub outputs_read: AtomicU64,
    pub last_output_ns: AtomicU64,
    /// 1 once the runtime has been closed.
    pub closed: AtomicU64,
    /// One stage's trace replay call (every stage adds one per step).
    pub dispatch: Timer,
    pub packet: Timer,
    pub hash: Timer,
    /// One Engram layer's rows for a step, gathered and written to every rank (every layer adds one per step).
    pub engram: Timer,
    pub output_read: Timer,
}

/// [`Counters`] in a `MAP_SHARED` mapping: of a new file put at `path`, or anonymous.
pub struct CounterMap {
    ptr: *mut Counters,
}

unsafe impl Send for CounterMap {}
unsafe impl Sync for CounterMap {}

impl CounterMap {
    pub fn open(path: Option<&str>) -> Result<Self, String> {
        let size = std::mem::size_of::<Counters>();
        // The file is built under another name and renamed over `path` once initialized: truncating a file a
        // monitor has mapped would fault (SIGBUS) its next read, whereas a replaced one stays mapped and valid.
        let staged = path.map(|p| format!("{p}.{}.tmp", std::process::id()));
        let (fd, flags) = match &staged {
            Some(staged) => {
                let c_path = CString::new(staged.as_str()).map_err(|e| e.to_string())?;
                let fd = unsafe {
                    libc::open(c_path.as_ptr(), libc::O_RDWR | libc::O_CREAT | libc::O_TRUNC | libc::O_CLOEXEC, 0o644)
                };
                if fd < 0 {
                    return Err(format!("open {staged}: {}", std::io::Error::last_os_error()));
                }
                if unsafe { libc::ftruncate(fd, size as libc::off_t) } != 0 {
                    let e = std::io::Error::last_os_error();
                    unsafe { libc::close(fd) };
                    return Err(format!("size {staged}: {e}"));
                }
                (fd, libc::MAP_SHARED)
            }
            None => (-1, libc::MAP_SHARED | libc::MAP_ANONYMOUS),
        };
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE, flags, fd, 0) };
        let mapped = std::io::Error::last_os_error();
        if fd >= 0 {
            unsafe { libc::close(fd) };
        }
        if ptr == libc::MAP_FAILED {
            return Err(format!("map the counters: {mapped}"));
        }
        // A fresh mapping is zeroed, and zero is every atomic's valid "0".
        let map = Self { ptr: ptr.cast() };
        map.version.store(VERSION, Ordering::Relaxed);
        map.pid.store(std::process::id() as u64, Ordering::Relaxed);
        map.started_ns.store(unix_ns(), Ordering::Relaxed);
        map.magic.store(MAGIC, Ordering::Release);
        if let (Some(staged), Some(path)) = (&staged, path) {
            std::fs::rename(staged, path).map_err(|e| format!("rename {staged} to {path}: {e}"))?;
        }
        Ok(map)
    }
}

impl std::ops::Deref for CounterMap {
    type Target = Counters;

    fn deref(&self) -> &Counters {
        unsafe { &*self.ptr }
    }
}

impl Drop for CounterMap {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr.cast(), std::mem::size_of::<Counters>()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_holds_the_monitor_layout() {
        let path = std::env::temp_dir().join(format!("dsv41-counters-test-{}", std::process::id()));
        let map = CounterMap::open(path.to_str()).unwrap();
        map.outputs_read.store(7, Ordering::Relaxed);
        map.output_read.nanos.store(9, Ordering::Relaxed);
        let bytes = std::fs::read(&path).unwrap();
        let word = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
        assert_eq!(bytes.len(), 20 * 8);
        assert_eq!((word(0), word(1), word(2)), (MAGIC, VERSION, std::process::id() as u64));
        assert_eq!((word(7), word(19)), (7, 9));
        drop(map);
        std::fs::remove_file(path).unwrap();
    }
}
