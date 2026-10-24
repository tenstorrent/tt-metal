# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Runs the framework's tt-d-gen engine driver (bringup_fw/testing/dgen_prefill_driver.py) against the pinned engine.

    <tt-d-gen python 3.12> _dgen_driver_shim.py <dgen_prefill_driver.py> <plan.json> <out.json>

Standard library and tt_engine only (it runs under tt-d-gen's interpreter, like the driver). From tt-d-gen #1600
(1cf34388) on, the engine reads completions_per_chunk from the runner's layer-completion ring header and the binding
is ``device_prefill_pipeline(service_id, ack_shm_name, chunk_size, connect_timeout_ms, sp_factor, num_mtp_tokens)``
(bindings/python/src/engine_ext.cpp:281-292). The driver still passes layers_per_chunk positionally after chunk_size,
which would land in connect_timeout_ms / sp_factor. This shim drops it when the binding has no such parameter and
leaves every other call of the driver as is. The engine's own work (slots, chunks, reuse, acks) is untouched.
"""

import runpy
import sys

import tt_engine as te

_orig = te.device_prefill_pipeline
_sig = (_orig.__doc__ or "").splitlines()[0] if _orig.__doc__ else ""
if "layers_per_chunk" not in _sig and "connect_timeout_ms" in _sig:

    def _compat(service_id, ack_shm_name, chunk_size, layers_per_chunk, connect_timeout_ms=30000, sp_factor=1):
        print(
            f"[dgen shim] the engine reads completions_per_chunk from the ring; layers_per_chunk "
            f"{layers_per_chunk} not passed",
            flush=True,
        )
        return _orig(service_id, ack_shm_name, chunk_size, int(connect_timeout_ms), int(sp_factor))

    te.device_prefill_pipeline = _compat

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        cfg = te.device_prefill_pipeline("svc", "/tt_prefill_layer_completions_svc", 5120, 6, 600000, 8)
        print(f"[dgen shim] {_sig} -> {type(cfg).__name__}")
        sys.exit(0)
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(5)
    driver = sys.argv[1]
    sys.argv = [driver, sys.argv[2], sys.argv[3]]
    runpy.run_path(driver, run_name="__main__")
