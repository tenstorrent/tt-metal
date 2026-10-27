// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// CPython extension module `_planar_concat`. Written against the bare CPython
// and NumPy C APIs so build.sh needs no dependency beyond a compiler and the
// NumPy headers already required to import the caller.

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <string>
#include <vector>

#include "planar_concat.hpp"

namespace {

using tt_dit_planar::DimOrder;
using tt_dit_planar::ShardView;

struct ComponentShards {
    std::vector<ShardView> views;
    int h_per = 0;
    int w_per = 0;
    int T = 0;
};

// Shards arrive pre-sorted by (mesh_row, mesh_col), so list order alone fixes
// each shard's placement in the mesh.
bool parse_component(PyObject* seq, DimOrder dim_order, int TP, int SP, const char* name, ComponentShards* out) {
    PyObject* fast = PySequence_Fast(seq, "shard container must be a sequence");
    if (fast == nullptr) {
        return false;
    }
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n != static_cast<Py_ssize_t>(TP) * SP) {
        PyErr_Format(PyExc_ValueError, "%s: expected %d shards for a %dx%d mesh, got %zd", name, TP * SP, TP, SP, n);
        Py_DECREF(fast);
        return false;
    }

    out->views.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(fast, i);
        if (!PyArray_Check(item)) {
            PyErr_Format(PyExc_TypeError, "%s[%zd]: expected a numpy.ndarray", name, i);
            Py_DECREF(fast);
            return false;
        }
        PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(item);
        if (PyArray_TYPE(arr) != NPY_UINT8 || !PyArray_IS_C_CONTIGUOUS(arr) || PyArray_NDIM(arr) != 4) {
            PyErr_Format(PyExc_ValueError, "%s[%zd]: expected a C-contiguous 4-D uint8 array", name, i);
            Py_DECREF(fast);
            return false;
        }

        int h_per, w_per, T;
        if (dim_order == DimOrder::CHWT) {
            h_per = static_cast<int>(PyArray_DIM(arr, 1));
            w_per = static_cast<int>(PyArray_DIM(arr, 2));
            T = static_cast<int>(PyArray_DIM(arr, 3));
        } else {
            T = static_cast<int>(PyArray_DIM(arr, 1));
            h_per = static_cast<int>(PyArray_DIM(arr, 2));
            w_per = static_cast<int>(PyArray_DIM(arr, 3));
        }
        if (i == 0) {
            out->h_per = h_per;
            out->w_per = w_per;
            out->T = T;
        } else if (h_per != out->h_per || w_per != out->w_per || T != out->T) {
            PyErr_Format(PyExc_ValueError, "%s[%zd]: shard shape differs from shard 0", name, i);
            Py_DECREF(fast);
            return false;
        }

        ShardView sv;
        sv.data = static_cast<const uint8_t*>(PyArray_DATA(arr));
        sv.r = static_cast<int>(i) / SP;
        sv.c = static_cast<int>(i) % SP;
        out->views.push_back(sv);
    }
    Py_DECREF(fast);
    return true;
}

PyObject* py_planar_concat(PyObject* /*self*/, PyObject* args) {
    PyObject *y_obj, *u_obj, *v_obj, *mesh_obj, *out_obj;
    const char* dim_order_str;
    int out_H, out_W;
    if (!PyArg_ParseTuple(
            args, "OOOsOOii", &y_obj, &u_obj, &v_obj, &dim_order_str, &mesh_obj, &out_obj, &out_H, &out_W)) {
        return nullptr;
    }

    DimOrder dim_order;
    if (std::string(dim_order_str) == "CHWT") {
        dim_order = DimOrder::CHWT;
    } else if (std::string(dim_order_str) == "CTHW") {
        dim_order = DimOrder::CTHW;
    } else {
        PyErr_Format(PyExc_ValueError, "dim_order must be 'CHWT' or 'CTHW', got '%s'", dim_order_str);
        return nullptr;
    }

    int TP, SP;
    {
        PyObject* fast = PySequence_Fast(mesh_obj, "mesh_shape must be a sequence");
        if (fast == nullptr) {
            return nullptr;
        }
        if (PySequence_Fast_GET_SIZE(fast) != 2) {
            PyErr_SetString(PyExc_ValueError, "mesh_shape must be a 2-element sequence");
            Py_DECREF(fast);
            return nullptr;
        }
        TP = static_cast<int>(PyLong_AsLong(PySequence_Fast_GET_ITEM(fast, 0)));
        SP = static_cast<int>(PyLong_AsLong(PySequence_Fast_GET_ITEM(fast, 1)));
        Py_DECREF(fast);
        if (PyErr_Occurred()) {
            return nullptr;
        }
    }
    if (TP <= 0 || SP <= 0) {
        PyErr_SetString(PyExc_ValueError, "mesh_shape entries must be positive");
        return nullptr;
    }

    ComponentShards y, cb, cr;
    if (!parse_component(y_obj, dim_order, TP, SP, "y_shards", &y) ||
        !parse_component(u_obj, dim_order, TP, SP, "u_shards", &cb) ||
        !parse_component(v_obj, dim_order, TP, SP, "v_shards", &cr)) {
        return nullptr;
    }
    if (cb.h_per != cr.h_per || cb.w_per != cr.w_per) {
        PyErr_SetString(PyExc_ValueError, "Cb and Cr shard dims differ");
        return nullptr;
    }
    if (y.T != cb.T || y.T != cr.T) {
        PyErr_SetString(PyExc_ValueError, "Y and chroma shards disagree on T");
        return nullptr;
    }

    if (!PyArray_Check(out_obj)) {
        PyErr_SetString(PyExc_TypeError, "out must be a numpy.ndarray");
        return nullptr;
    }
    PyArrayObject* out_arr = reinterpret_cast<PyArrayObject*>(out_obj);
    if (PyArray_TYPE(out_arr) != NPY_UINT8 || !PyArray_IS_C_CONTIGUOUS(out_arr) || PyArray_NDIM(out_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "out must be a C-contiguous 2-D uint8 array");
        return nullptr;
    }
    if (out_H <= 0 || out_W <= 0 || (out_H & 1) || (out_W & 1)) {
        PyErr_Format(PyExc_ValueError, "out_H/out_W must be positive and even, got %d/%d", out_H, out_W);
        return nullptr;
    }

    const int H = y.h_per * TP;
    const int W = y.w_per * SP;
    if (out_H > H || out_W > W) {
        PyErr_Format(PyExc_ValueError, "logical crop %dx%d exceeds the assembled %dx%d frame", out_H, out_W, H, W);
        return nullptr;
    }

    const npy_intp row_stride =
        static_cast<npy_intp>(out_H) * out_W + 2 * (static_cast<npy_intp>(out_H / 2) * (out_W / 2));
    if (PyArray_DIM(out_arr, 0) != y.T || PyArray_DIM(out_arr, 1) != row_stride) {
        PyErr_Format(
            PyExc_ValueError,
            "out must have shape (%d, %zd); got (%zd, %zd)",
            y.T,
            row_stride,
            PyArray_DIM(out_arr, 0),
            PyArray_DIM(out_arr, 1));
        return nullptr;
    }

    uint8_t* out_ptr = static_cast<uint8_t*>(PyArray_DATA(out_arr));
    Py_BEGIN_ALLOW_THREADS;
    tt_dit_planar::planar_concat(
        y.views, y.h_per, y.w_per, cb.views, cb.h_per, cb.w_per, cr.views, dim_order, y.T, H, W, out_H, out_W, out_ptr);
    Py_END_ALLOW_THREADS;

    Py_RETURN_NONE;
}

PyObject* py_set_thread_pool_size(PyObject* /*self*/, PyObject* arg) {
    const long n = PyLong_AsLong(arg);
    if (n == -1 && PyErr_Occurred()) {
        return nullptr;
    }
    tt_dit_planar::set_thread_pool_size(static_cast<int>(n));
    Py_RETURN_NONE;
}

PyMethodDef g_methods[] = {
    {"planar_concat", py_planar_concat, METH_VARARGS, "Scatter YUV 4:2:0 shards into a planar frame buffer."},
    {"set_thread_pool_size",
     py_set_thread_pool_size,
     METH_O,
     "Set the worker pool size; takes effect before first use."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef g_module = {
    PyModuleDef_HEAD_INIT,
    "_planar_concat",
    "AVX2 planar YUV concat.",
    -1,
    g_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__planar_concat(void) {
    import_array();
    return PyModule_Create(&g_module);
}
