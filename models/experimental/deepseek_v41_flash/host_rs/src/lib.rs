//! `dsv41_host`: DeepSeek V4.1's host runtime as a Python extension. Python builds the model and captures the
//! decode traces; this module then runs the per-step host work on native threads (see `decode`), and does the
//! Engram host lookup for decode and prefill alike (see `engram`). Python-facing wrappers live in
//! `models/experimental/deepseek_v41_flash/tt/host_rs.py`.

pub mod counters;
pub mod decode;
pub mod engram;
pub mod ffi;

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;

use decode::{DecodeConfig, DecodeHost, EngramSockets};
use engram::{Engram, EngramArgs, Table};
use ffi::{MeshDevice, D2H, H2D};

fn err(message: String) -> PyErr {
    PyRuntimeError::new_err(message)
}

fn u16_bytes<'py>(py: Python<'py>, values: &[u16]) -> Bound<'py, PyByteArray> {
    PyByteArray::new_bound(py, unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() * 2) })
}

/// Load the `ttnn_hr_*` C ABI from `_ttnn.so` (pass `ttnn._ttnn.__file__`).
#[pyfunction]
fn load_ttnn(path: &str) -> PyResult<()> {
    ffi::load(path).map(|_| ()).map_err(err)
}

/// The Engram host lookup (`EngramHostLookup`'s arithmetic), shared by decode's Engram threads and prefill.
#[pyclass(frozen, name = "Engram", module = "dsv41_host")]
struct PyEngram {
    inner: Arc<Engram>,
}

#[pymethods]
impl PyEngram {
    /// `tables` is `[(layer_id, weight_ptr, scale_ptr, rows)]`: the tables' memory, which the caller keeps alive.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (token_map, vocab_size, pad_id, max_ngram, n_heads, head_dim, layout_layers, multipliers,
                        primes, offsets, tables, batch, max_seq, sessions))]
    fn new(
        token_map: Vec<i64>,
        vocab_size: i64,
        pad_id: i64,
        max_ngram: usize,
        n_heads: usize,
        head_dim: usize,
        layout_layers: Vec<i64>,
        multipliers: Vec<Vec<i64>>,
        primes: Vec<Vec<i64>>,
        offsets: Vec<Vec<i64>>,
        tables: Vec<(i64, usize, usize, usize)>,
        batch: usize,
        max_seq: usize,
        sessions: usize,
    ) -> PyResult<Self> {
        let tables = tables
            .into_iter()
            .map(|(layer, weight, scale, rows)| {
                (layer, Table { weight: weight as *const u8, scale: scale as *const u8, rows })
            })
            .collect();
        let inner = Engram::new(EngramArgs {
            token_map,
            vocab_size,
            pad_id,
            max_ngram,
            n_heads,
            head_dim,
            layout_layers,
            multipliers,
            primes,
            offsets,
            tables,
            batch,
            max_seq,
            sessions,
        })
        .map_err(err)?;
        Ok(Self { inner: Arc::new(inner) })
    }

    #[getter]
    fn kin(&self) -> usize {
        self.inner.kin()
    }

    #[getter]
    fn n_cols(&self) -> usize {
        self.inner.n_cols()
    }

    #[getter]
    fn table_layers(&self) -> Vec<i64> {
        self.inner.table_layers().to_vec()
    }

    fn use_session(&self, slot: usize) -> PyResult<()> {
        self.inner.use_session(slot).map_err(err)
    }

    fn export_history(&self, slot: usize, pos: usize) -> PyResult<Vec<i64>> {
        self.inner.export_history(slot, pos).map_err(err)
    }

    fn import_history(&self, slot: usize, values: Vec<i64>, length: usize) -> PyResult<()> {
        self.inner.import_history(slot, &values, length).map_err(err)
    }

    /// Hashes `[batch, len, n_layout_layers, n_cols]` (flat) of `ids` (`[batch, len]`, flat) at `start` against
    /// slot `slot`'s history, which it extends.
    fn hash(&self, py: Python<'_>, slot: usize, ids: Vec<i64>, start: usize) -> PyResult<Vec<i64>> {
        let inner = &self.inner;
        py.allow_threads(|| inner.hash(slot, &ids, start)).map_err(err)
    }

    /// Rows `ids` of layer `layer_id`'s table, dequantized: bf16 bytes `[len(ids), head_dim]`.
    fn gather<'py>(&self, py: Python<'py>, layer_id: i64, ids: Vec<i64>) -> PyResult<Bound<'py, PyByteArray>> {
        let inner = &self.inner;
        let t = inner
            .table_layers()
            .iter()
            .position(|&l| l == layer_id)
            .ok_or_else(|| err(format!("no table for layer {layer_id}")))?;
        let hd = inner.kin() / inner.n_cols();
        let out = py
            .allow_threads(|| {
                let mut out = vec![0u16; ids.len() * hd];
                inner.gather(t, &ids, &mut out).map(|_| out)
            })
            .map_err(err)?;
        Ok(u16_bytes(py, &out))
    }

    /// `EngramHostLookup.__call__` on the current session: each table's rows, bf16 `[batch, len, kin]`, written
    /// into the contiguous buffers at `outs` (one per table, in table order; e.g. torch tensors' `data_ptr()`).
    fn rows_into(&self, py: Python<'_>, ids: Vec<i64>, start: usize, outs: Vec<usize>) -> PyResult<()> {
        let inner = &self.inner;
        let n = ids.len() * inner.kin();
        py.allow_threads(|| {
            let mut buffers: Vec<&mut [u16]> =
                outs.iter().map(|&p| unsafe { std::slice::from_raw_parts_mut(p as *mut u16, n) }).collect();
            inner.rows_into(&ids, start, &mut buffers)
        })
        .map_err(err)
    }
}

/// The traced decode step's host side (see `decode`).
#[pyclass(frozen, name = "DecodeHost", module = "dsv41_host")]
struct PyDecodeHost {
    inner: DecodeHost,
}

fn h2d(handle: &Bound<'_, PyAny>) -> PyResult<H2D> {
    if let Ok(address) = handle.extract::<usize>() {
        return H2D::borrowed(address).map_err(err);
    }
    let name: String = handle.extract()?;
    H2D::connect(&name, 0).map_err(err)
}

fn d2h(handle: &Bound<'_, PyAny>) -> PyResult<D2H> {
    if let Ok(address) = handle.extract::<usize>() {
        return D2H::borrowed(address).map_err(err);
    }
    let name: String = handle.extract()?;
    D2H::connect(&name, 0).map_err(err)
}

#[pymethods]
impl PyDecodeHost {
    /// A socket argument is the address of Python's socket (`native_handle`) or the id of a named socket
    /// (`export_descriptor`), which this runtime connects to. `engram_sockets` is
    /// `[(layer_id, [socket per TP rank], local_kin)]`. `counters_path` is a file (in `/dev/shm`) to keep the
    /// runtime's counters in, for the web monitor.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (stages, variants, packet_socket, batch, sliding_window, win_slot_rows, output_socket,
                        output_step_bytes, engram=None, engram_sockets=Vec::new(), cpus=Vec::new(),
                        counters_path=None))]
    fn new(
        stages: Vec<usize>,
        variants: Vec<Vec<u32>>,
        packet_socket: &Bound<'_, PyAny>,
        batch: usize,
        sliding_window: i64,
        win_slot_rows: i64,
        output_socket: &Bound<'_, PyAny>,
        output_step_bytes: usize,
        engram: Option<&Bound<'_, PyEngram>>,
        engram_sockets: Vec<(i64, Vec<Bound<'_, PyAny>>, usize)>,
        cpus: Vec<usize>,
        counters_path: Option<String>,
    ) -> PyResult<Self> {
        let engram = engram.map(|e| e.get().inner.clone());
        let mut layers = Vec::new();
        for (layer_id, ranks, local_kin) in &engram_sockets {
            let table = engram
                .as_ref()
                .and_then(|e| e.table_layers().iter().position(|l| l == layer_id))
                .ok_or_else(|| err(format!("no Engram table for layer {layer_id}")))?;
            let ranks = ranks.iter().map(h2d).collect::<PyResult<Vec<_>>>()?;
            layers.push(EngramSockets { table, ranks, local_kin: *local_kin });
        }
        let config = DecodeConfig {
            stages: stages.into_iter().map(|a| MeshDevice(a as *mut _)).collect(),
            variants,
            packet: h2d(packet_socket)?,
            batch,
            sliding_window,
            win_slot_rows,
            engram,
            engram_sockets: layers,
            output: d2h(output_socket)?,
            output_step_bytes,
            cpus,
            counters_path,
        };
        Ok(Self { inner: DecodeHost::start(config).map_err(err)? })
    }

    fn dispatch(&self, variant: usize) -> PyResult<()> {
        self.inner.dispatch(variant).map_err(err)
    }

    fn write_step(&self, tokens: Vec<i64>, pos: i64, slot: i64) -> PyResult<()> {
        self.inner.write_step(tokens, pos, slot).map_err(err)
    }

    fn drain(&self, py: Python<'_>) -> PyResult<()> {
        let inner = &self.inner;
        py.allow_threads(|| inner.drain()).map_err(err)
    }

    fn wait_written(&self, py: Python<'_>) -> PyResult<()> {
        let inner = &self.inner;
        py.allow_threads(|| inner.wait_written()).map_err(err)
    }

    fn read_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyByteArray>> {
        let inner = &self.inner;
        let out = py.allow_threads(|| inner.read_output()).map_err(err)?;
        Ok(PyByteArray::new_bound(py, &out))
    }

    fn outputs_ready(&self) -> usize {
        self.inner.outputs_ready()
    }

    /// `{name: (count, mean microseconds)}` of the threads' work so far.
    fn stats(&self) -> std::collections::HashMap<&'static str, (u64, f64)> {
        self.inner.stats().into_iter().map(|(name, count, mean)| (name, (count, mean))).collect()
    }

    fn close(&self, py: Python<'_>) {
        let inner = &self.inner;
        py.allow_threads(|| inner.close());
    }
}

#[pymodule]
fn dsv41_host(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_ttnn, m)?)?;
    m.add_class::<PyEngram>()?;
    m.add_class::<PyDecodeHost>()?;
    Ok(())
}
