# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn


@pytest.fixture(autouse=True)
def _isolate_fft_device_state(device):
    """Give every FFT test a clean device-side view.

    Three sources of cross-test state can leak on the module-scoped `device`
    fixture and produce spurious failures whose parameter is not the root
    cause:

    * ttnn's per-device program cache carries entries from earlier tests
      in the same session.  A cached ProgramSpec entry keyed on
      (shape, dtype, has_imag) reuses whatever runtime args were baked in
      by the first caller, which is only safe when the second caller's
      tensor addresses / plan buffers match.  Clearing the cache before
      each test forces a fresh factory dispatch.
    * In-flight NoC transactions from the previous test's final dispatch
      can race the next test's tensor allocations if the allocator hands
      out the same L1 / DRAM address before those transactions drain.
      Synchronising the device *after* each test forces a full drain
      before the next test starts (see the Bluestein-small intermittent
      failure noted in the PR body).
    * The FFT-specific host plan caches (Bluestein, Stockham twiddles,
      apply_twiddles delta tables, ...) own device tensors whose
      destructors race GraphTracker teardown at process exit.  Releasing
      them after each test reclaims device memory and shortens exit.

    All three clears are best-effort: the `getattr` guards make the
    fixture inert on older builds that predate the C++ bindings.

    Known limitation: even with these mitigations, a small subset of the
    multi-pass tests (`test_fft_two_pass`, `test_fft_three_pass`,
    `test_fft_radix_pass_native`, and a few large-N `test_fft_all_n`
    parametrisations) can still fail when the whole directory is run in
    one pytest session, because pytest's session-scoped device fixture
    outlives all of the above teardown hooks and some LLK-level state
    (unpacker/packer datatype registers, tile-dim reconfig cache) is not
    reachable from the Python API.  Every affected test passes in
    isolation and per-file.  For CI or a green full-suite run, invoke
    with `pytest --forked`, which puts each test in its own subprocess
    and eliminates the shared state entirely.
    """
    clear_program_cache = getattr(device, "clear_program_cache", None)
    if clear_program_cache is not None:
        clear_program_cache()

    yield

    synchronize_device = getattr(ttnn, "synchronize_device", None)
    if synchronize_device is not None:
        try:
            synchronize_device(device)
        except Exception:
            # A test that already crashed the device shouldn't mask its
            # own failure with a teardown error.
            pass

    clear_fft = getattr(ttnn.experimental, "clear_fft_device_caches", None)
    if clear_fft is not None:
        clear_fft()
