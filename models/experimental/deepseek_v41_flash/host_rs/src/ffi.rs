//! The `ttnn_hr_*` C ABI of `_ttnn.so` (`ttnn/cpp/ttnn-nanobind/host_runtime.cpp`): trace replay and the
//! H2D / D2H sockets, called on the C++ objects Python built (by address) or on named-socket connectors.

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::OnceLock;

use libloading::Library;

const ABI_VERSION: u32 = 1;
const ERR_LEN: usize = 1024;

type ExecuteTrace = unsafe extern "C" fn(*mut c_void, u32, u8, *mut c_char, usize) -> i32;
type PageSize = unsafe extern "C" fn(*mut c_void) -> u32;
type HasSpace = unsafe extern "C" fn(*mut c_void, u32, *mut c_char, usize) -> i32;
type Write = unsafe extern "C" fn(*mut c_void, *const c_void, u32, *mut c_char, usize) -> i32;
type PagesAvailable = unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> i32;
type Read = unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *mut c_char, usize) -> i32;
type Connect = unsafe extern "C" fn(*const c_char, u32, *mut c_char, usize) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);

pub struct Ttnn {
    _lib: Library,
    execute_trace: ExecuteTrace,
    h2d_page_size: PageSize,
    h2d_has_space: HasSpace,
    h2d_write: Write,
    h2d_connect: Connect,
    h2d_destroy: Destroy,
    d2h_page_size: PageSize,
    d2h_pages_available: PagesAvailable,
    d2h_read: Read,
    d2h_connect: Connect,
    d2h_destroy: Destroy,
}

static TTNN: OnceLock<Ttnn> = OnceLock::new();

/// Load the C ABI from `path` (`ttnn._ttnn.__file__`, already loaded by `import ttnn`, so this only finds it).
pub fn load(path: &str) -> Result<&'static Ttnn, String> {
    if let Some(ttnn) = TTNN.get() {
        return Ok(ttnn);
    }
    let ttnn = unsafe { Ttnn::open(path) }.map_err(|e| format!("loading the ttnn host runtime ABI from {path}: {e}"))?;
    Ok(TTNN.get_or_init(|| ttnn))
}

pub fn get() -> Result<&'static Ttnn, String> {
    TTNN.get().ok_or_else(|| "the ttnn host runtime ABI is not loaded".to_string())
}

impl Ttnn {
    unsafe fn open(path: &str) -> Result<Self, libloading::Error> {
        let lib = Library::new(path)?;
        let version: libloading::Symbol<unsafe extern "C" fn() -> u32> = lib.get(b"ttnn_hr_abi_version\0")?;
        let found = version();
        if found != ABI_VERSION {
            panic!("ttnn host runtime ABI version {found}, this crate needs {ABI_VERSION}: rebuild one of them");
        }
        Ok(Self {
            execute_trace: *lib.get(b"ttnn_hr_execute_trace\0")?,
            h2d_page_size: *lib.get(b"ttnn_hr_h2d_page_size\0")?,
            h2d_has_space: *lib.get(b"ttnn_hr_h2d_has_space\0")?,
            h2d_write: *lib.get(b"ttnn_hr_h2d_write\0")?,
            h2d_connect: *lib.get(b"ttnn_hr_h2d_connect\0")?,
            h2d_destroy: *lib.get(b"ttnn_hr_h2d_destroy\0")?,
            d2h_page_size: *lib.get(b"ttnn_hr_d2h_page_size\0")?,
            d2h_pages_available: *lib.get(b"ttnn_hr_d2h_pages_available\0")?,
            d2h_read: *lib.get(b"ttnn_hr_d2h_read\0")?,
            d2h_connect: *lib.get(b"ttnn_hr_d2h_connect\0")?,
            d2h_destroy: *lib.get(b"ttnn_hr_d2h_destroy\0")?,
            _lib: lib,
        })
    }
}

fn checked(rc: i32, err: &[u8; ERR_LEN], what: &str) -> Result<i32, String> {
    if rc >= 0 {
        return Ok(rc);
    }
    let message = CStr::from_bytes_until_nul(err).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    Err(format!("{what}: {message}"))
}

/// A `MeshDevice*` (one pipeline stage's submesh) owned by Python.
#[derive(Clone, Copy)]
pub struct MeshDevice(pub *mut c_void);
// The C++ object is internally synchronized per command queue; one dispatch thread owns each stage.
unsafe impl Send for MeshDevice {}
unsafe impl Sync for MeshDevice {}

impl MeshDevice {
    pub fn execute_trace(&self, trace_id: u32) -> Result<(), String> {
        let ttnn = get()?;
        let mut err = [0u8; ERR_LEN];
        let rc = unsafe { (ttnn.execute_trace)(self.0, trace_id, 0, err.as_mut_ptr().cast(), ERR_LEN) };
        checked(rc, &err, "execute_trace").map(|_| ())
    }
}

/// An `H2DSocket*`: Python's own object (`owned == false`) or a named-socket connector this crate frees.
pub struct H2D {
    ptr: *mut c_void,
    owned: bool,
    page_size: usize,
}
// A socket FIFO is single-producer: exactly one thread writes each H2D at a time (its writer thread).
unsafe impl Send for H2D {}
unsafe impl Sync for H2D {}

impl H2D {
    pub fn borrowed(address: usize) -> Result<Self, String> {
        let ptr = address as *mut c_void;
        let page_size = unsafe { (get()?.h2d_page_size)(ptr) } as usize;
        if page_size == 0 {
            return Err("H2D socket has no page size set".into());
        }
        Ok(Self { ptr, owned: false, page_size })
    }

    /// Attach to the socket exported as `socket_id` (`H2DSocket.export_descriptor`), resuming its counters.
    pub fn connect(socket_id: &str, timeout_ms: u32) -> Result<Self, String> {
        let ttnn = get()?;
        let id = CString::new(socket_id).map_err(|e| e.to_string())?;
        let mut err = [0u8; ERR_LEN];
        let ptr = unsafe { (ttnn.h2d_connect)(id.as_ptr(), timeout_ms, err.as_mut_ptr().cast(), ERR_LEN) };
        if ptr.is_null() {
            checked(-1, &err, &format!("connecting H2D socket {socket_id}"))?;
        }
        let page_size = unsafe { (ttnn.h2d_page_size)(ptr) } as usize;
        Ok(Self { ptr, owned: true, page_size })
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn has_space(&self, num_bytes: usize) -> Result<bool, String> {
        let mut err = [0u8; ERR_LEN];
        let rc = unsafe { (get()?.h2d_has_space)(self.ptr, num_bytes as u32, err.as_mut_ptr().cast(), ERR_LEN) };
        checked(rc, &err, "H2D has_space").map(|v| v == 1)
    }

    /// Write whole pages; blocks while the FIFO is full.
    pub fn write(&self, data: &[u8]) -> Result<(), String> {
        if data.is_empty() || data.len() % self.page_size != 0 {
            return Err(format!("H2D write of {} B is not whole {} B pages", data.len(), self.page_size));
        }
        let pages = (data.len() / self.page_size) as u32;
        let mut err = [0u8; ERR_LEN];
        let rc = unsafe { (get()?.h2d_write)(self.ptr, data.as_ptr().cast(), pages, err.as_mut_ptr().cast(), ERR_LEN) };
        checked(rc, &err, "H2D write").map(|_| ())
    }
}

impl Drop for H2D {
    fn drop(&mut self) {
        if self.owned {
            if let Ok(ttnn) = get() {
                unsafe { (ttnn.h2d_destroy)(self.ptr) };
            }
        }
    }
}

/// A `D2HSocket*`, as [`H2D`].
pub struct D2H {
    ptr: *mut c_void,
    owned: bool,
    page_size: usize,
}
unsafe impl Send for D2H {}
unsafe impl Sync for D2H {}

impl D2H {
    pub fn borrowed(address: usize) -> Result<Self, String> {
        let ptr = address as *mut c_void;
        let page_size = unsafe { (get()?.d2h_page_size)(ptr) } as usize;
        if page_size == 0 {
            return Err("D2H socket has no page size set".into());
        }
        Ok(Self { ptr, owned: false, page_size })
    }

    pub fn connect(socket_id: &str, timeout_ms: u32) -> Result<Self, String> {
        let ttnn = get()?;
        let id = CString::new(socket_id).map_err(|e| e.to_string())?;
        let mut err = [0u8; ERR_LEN];
        let ptr = unsafe { (ttnn.d2h_connect)(id.as_ptr(), timeout_ms, err.as_mut_ptr().cast(), ERR_LEN) };
        if ptr.is_null() {
            checked(-1, &err, &format!("connecting D2H socket {socket_id}"))?;
        }
        let page_size = unsafe { (ttnn.d2h_page_size)(ptr) } as usize;
        Ok(Self { ptr, owned: true, page_size })
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn pages_available(&self) -> Result<usize, String> {
        let mut err = [0u8; ERR_LEN];
        let rc = unsafe { (get()?.d2h_pages_available)(self.ptr, err.as_mut_ptr().cast(), ERR_LEN) };
        checked(rc, &err, "D2H pages_available").map(|v| v as usize)
    }

    /// Read whole pages into `out`; blocks until they are in.
    pub fn read(&self, out: &mut [u8]) -> Result<(), String> {
        if out.is_empty() || out.len() % self.page_size != 0 {
            return Err(format!("D2H read of {} B is not whole {} B pages", out.len(), self.page_size));
        }
        let pages = (out.len() / self.page_size) as u32;
        let mut err = [0u8; ERR_LEN];
        let rc =
            unsafe { (get()?.d2h_read)(self.ptr, out.as_mut_ptr().cast(), pages, err.as_mut_ptr().cast(), ERR_LEN) };
        checked(rc, &err, "D2H read").map(|_| ())
    }
}

impl Drop for D2H {
    fn drop(&mut self) {
        if self.owned {
            if let Ok(ttnn) = get() {
                unsafe { (ttnn.d2h_destroy)(self.ptr) };
            }
        }
    }
}
