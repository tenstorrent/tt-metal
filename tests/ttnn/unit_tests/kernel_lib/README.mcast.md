# Multicast helper validation

Activate the repository's `python_env` and run device tests sequentially through
`scripts/run_safe_pytest.sh`. Start with one device smoke case:

```bash
scripts/run_safe_pytest.sh --no-precompile tests/ttnn/unit_tests/kernel_lib/test_mcast_wrappers.py::test_smoke
```

The multicast Python suites are `test_mcast_wrappers.py`, `test_mcast_family.py`,
`test_mcast_contracts.py`, `test_mcast_raw_pipe.py`, `test_chain_signal_stress.py`, and
`test_chain_write_ordering.py`. Python API tests live in
`tests/ttnn/unit_tests/base_functionality/test_mcast_{host,spec}_bindings.py`.

Build native tests with `./build_metal.sh --build-ttnn-tests`, then run their launcher
in a **separate pytest invocation**:

```bash
scripts/run_safe_pytest.sh --no-precompile tests/ttnn/unit_tests/kernel_lib/mcast_native_tests.py
```

This runs the native host contracts, existing GroupNorm geometry regressions, and
ProgramSpec device smoke/matrix tests. `mcast_native_tests.py` deliberately avoids default
pytest filename patterns: a Python process retains its chip driver after closing a device,
so starting a native child after Python device tests would leave the child waiting for
its parent's chip lock.

The migrated consumer checks are the `toy_spec_mcast` and `toy_variance` operation tests,
and the `mcast_topology`, `shared_input_reuse`, and `tensix_all_reduce` example tests.
