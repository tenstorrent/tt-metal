"""Pytest plugin that installs the CPU-compat shims into the test subprocess.

The orchestrator installs CPU-compat (mamba-ssm provider with a real pure-torch
rmsnorm_fn, cuda-stream neutralizer, accelerator-package hollow stubs) via
install_cpu_compat() before it loads any HF reference model. The PCC tests run
in a SEPARATE pytest subprocess that does not inherit those shims, so an HF
model with a hard mamba-ssm import gate (or an unguarded torch.cuda.stream)
fails to load and every component test errors identically.

Loading this plugin via ``pytest -p scripts.tt_hw_planner.cpu_compat_pytest_plugin``
runs the SAME install_cpu_compat() at plugin-registration time — before test
collection and before any from_pretrained — so every component test inherits
the exact shims the orchestrator uses. Generic across models (install_cpu_compat
decides what is needed), deterministic, and reuses the committed intelligence
instead of a per-demo hand-rolled conftest shim.

Import-time failures are swallowed so the plugin can never break pytest startup
on an environment where CPU-compat is unavailable or unnecessary.
"""

try:
    from .cpu_compat import install_cpu_compat as _install_cpu_compat

    _install_cpu_compat()
except Exception:
    pass
