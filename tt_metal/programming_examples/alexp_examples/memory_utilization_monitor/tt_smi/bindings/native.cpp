// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../tt_smi_backend.hpp"

namespace py = pybind11;

PYBIND11_MODULE(native, m) {
    m.doc() = "TT-SMI native backend bindings";

    py::class_<tt_smi::TelemetryData>(m, "TelemetryData")
        .def(py::init<>())
        .def_readwrite("temperature", &tt_smi::TelemetryData::temperature)
        .def_readwrite("power", &tt_smi::TelemetryData::power)
        .def_readwrite("voltage_mv", &tt_smi::TelemetryData::voltage_mv)
        .def_readwrite("current_ma", &tt_smi::TelemetryData::current_ma)
        .def_readwrite("aiclk_mhz", &tt_smi::TelemetryData::aiclk_mhz)
        .def_readwrite("status", &tt_smi::TelemetryData::status)
        .def_readwrite("available", &tt_smi::TelemetryData::available);

    py::class_<tt_smi::ProcessMemory>(m, "ProcessMemory")
        .def(py::init<>())
        .def_readwrite("pid", &tt_smi::ProcessMemory::pid)
        .def_readwrite("name", &tt_smi::ProcessMemory::name)
        .def_readwrite("dram_allocated", &tt_smi::ProcessMemory::dram_allocated)
        .def_readwrite("l1_allocated", &tt_smi::ProcessMemory::l1_allocated)
        .def_readwrite("l1_small_allocated", &tt_smi::ProcessMemory::l1_small_allocated)
        .def_readwrite("trace_allocated", &tt_smi::ProcessMemory::trace_allocated)
        .def_readwrite("cb_allocated", &tt_smi::ProcessMemory::cb_allocated)
        .def_readwrite("kernel_allocated", &tt_smi::ProcessMemory::kernel_allocated);

    py::class_<tt_smi::Device>(m, "Device")
        .def(py::init<>())
        .def_readwrite("chip_id", &tt_smi::Device::chip_id)
        .def_readwrite("asic_id", &tt_smi::Device::asic_id)
        .def_readwrite("arch_name", &tt_smi::Device::arch_name)
        .def_readwrite("is_remote", &tt_smi::Device::is_remote)
        .def_readwrite("tray_id", &tt_smi::Device::tray_id)
        .def_readwrite("chip_in_tray", &tt_smi::Device::chip_in_tray)
        .def_readwrite("asic_location", &tt_smi::Device::asic_location)
        .def_readwrite("display_id", &tt_smi::Device::display_id)
        .def_readwrite("telemetry", &tt_smi::Device::telemetry)
        .def_readwrite("total_dram", &tt_smi::Device::total_dram)
        .def_readwrite("used_dram", &tt_smi::Device::used_dram)
        .def_readwrite("total_l1", &tt_smi::Device::total_l1)
        .def_readwrite("used_l1", &tt_smi::Device::used_l1)
        .def_readwrite("used_l1_small", &tt_smi::Device::used_l1_small)
        .def_readwrite("used_trace", &tt_smi::Device::used_trace)
        .def_readwrite("used_cb", &tt_smi::Device::used_cb)
        .def_readwrite("used_kernel", &tt_smi::Device::used_kernel)
        .def_readwrite("processes", &tt_smi::Device::processes)
        .def_readwrite("has_shm", &tt_smi::Device::has_shm);

    m.def("enumerate_devices", &tt_smi::enumerate_devices, py::arg("shm_only") = false, "Enumerate all TT devices");

    m.def("update_device_telemetry", &tt_smi::update_device_telemetry, "Update telemetry for a device");

    m.def("update_device_memory", &tt_smi::update_device_memory, "Update memory stats for a device");

    m.def("cleanup_dead_processes", &tt_smi::cleanup_dead_processes, "Clean up dead processes from SHM");

    m.def("format_bytes", &tt_smi::format_bytes, "Format bytes with units (KiB, MiB, GiB)");

    m.def(
        "reset_devices",
        &tt_smi::reset_devices,
        py::arg("device_ids") = std::vector<int>{},
        py::arg("reset_m3") = false,
        "Reset TT devices. Empty device_ids resets ALL devices. reset_m3=True for deeper M3 board reset.");
}
