"""CPU-only admission of the exact built Flash source, followed by original host tests."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

root = Path(sys.argv[1]).resolve(strict=True)
out = Path(sys.argv[2]).resolve()
out.mkdir(exist_ok=False)
assert sys.version_info[:2] == (3, 12), sys.version
assert platform.machine() == "x86_64", platform.machine()


def tt_fds():
    found = {}
    for p in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(p)
        except FileNotFoundError:
            continue
        if target.startswith("/dev/tenstorrent/"):
            found[p.name] = target
    return found


assert not tt_fds()
import ttnn


def refuse_device(*args, **kwargs):
    raise AssertionError("CPU host admission must not open a Tenstorrent device")


# These host tests use mock device operations. Real imports must never open a device.
for name in ("open_device", "open_mesh_device", "CreateDevice", "CreateDevices"):
    if hasattr(ttnn, name):
        setattr(ttnn, name, refuse_device)

import models.demos.blackhole.qwen38_flash_next.ttnn.model
import models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm
import pytest

origins = {}
for name, module in tuple(sys.modules.items()):
    if not (name == "ttnn" or name.startswith(("ttnn.", "models."))):
        continue
    path = getattr(module, "__file__", None)
    if path:
        p = Path(path).resolve(strict=True)
        assert p.is_relative_to(root), (name, str(p), str(root))
        origins[name] = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
assert not tt_fds()
(out / "import-proof.json").write_text(
    json.dumps({"python": sys.version, "origins": origins, "tt_fds": tt_fds()}, indent=2) + "\n"
)
rc = pytest.main(
    [
        "--noconftest",
        "-q",
        str(root / "models/demos/blackhole/qwen38_flash_next/tests/test_moe_public_api_host.py"),
        str(root / "models/demos/blackhole/qwen38_flash_next/tests/test_ttnn_head_tail_split_static.py")
        + "::test_handoff_is_the_layer_0_residual_only_with_pinned_metadata",
        str(root / "models/demos/blackhole/qwen38_flash_next/tests/test_ttnn_head_tail_split_static.py")
        + "::test_handoff_does_not_retain_module_constants",
        str(root / "models/demos/blackhole/qwen38_flash_next/tests/test_ttnn_head_tail_split_static.py")
        + "::test_handoff_copy_replace_and_custom_metadata",
        str(root / "models/demos/blackhole/qwen38_flash_next/tests/test_qwen38_vllm_no_device.py")
        + "::test_bundle_metadata_names_the_hf_architecture_and_the_adapter",
        "--junitxml=" + str(out / "host.xml"),
    ]
)
assert not tt_fds()
(out / "host-result.json").write_text(
    json.dumps({"pytest_exit": rc, "tt_fds": tt_fds(), "completed_unix": time.time()}, indent=2) + "\n"
)
raise SystemExit(rc)
