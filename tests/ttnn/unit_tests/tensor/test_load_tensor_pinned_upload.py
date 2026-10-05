# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.load_tensor(file, device=...) of a tensor file above the pinned-write threshold must finish.

load_tensor maps the file read-only, and an upload above Metal's 32 MiB pinned-write threshold pins that
mapping device-read-only (PinnedMemoryCache::try_pin). With a MAP_PRIVATE mapping, a long-term read-only
pin makes the kernel unshare (copy) each file page first; on Linux 6.8 with a freshly written XFS file the
pin ioctl instead never returned: 100% system time, flat user time, and the system-wide thp_file_mapped and
thp_split_pmd counters rising together by ~450k/s (a page-cache PMD mapped and split again on every retry),
only SIGKILL-able, with the device open. A MAP_SHARED mapping is pinned in place and completes.

The file is written under the checkout's ignored generated/ directory, where model weight caches live, rather than
pytest's tmp_path: on the host where this was found, the same upload from a /tmp (overlayfs) file completed even
before the fix, while the checkout's XFS reproduced the hang every time.

Because a regression hangs the process rather than failing, the upload runs in a child process that a
default-action SIGALRM kills at a deadline; the test fails instead of wedging the runner and the device lock.
"""

import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

# 1024 * 9216 * 4 bytes = 36 MiB: one device's upload is above Metal's 32 MiB pinned-write threshold.
SHAPE = (1, 1, 1024, 9216)
# Device open and file write are outside this budget; the healthy load takes well under a second.
LOAD_DEADLINE_S = 30
CHILD_DEADLINE_S = 300
DONE_MARKER = "LOAD_TENSOR_PINNED_UPLOAD_OK"
GENERATED_DIR = Path(__file__).resolve().parents[4] / "generated"


def _child(file_name: str) -> None:
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        # Position-dependent values: a uniform buffer compares equal even if a wrong offset or page is transferred.
        torch_tensor = torch.arange(SHAPE[-2] * SHAPE[-1], dtype=torch.float32).reshape(SHAPE)
        ttnn.dump_tensor(file_name, ttnn.from_torch(torch_tensor, dtype=ttnn.float32))
        # SIGALRM has no Python handler, so its default action kills the process even from inside the stuck ioctl.
        signal.alarm(LOAD_DEADLINE_S)
        device_tensor = ttnn.load_tensor(file_name, device=device)
        ttnn.synchronize_device(device)
        signal.alarm(0)
        assert torch.equal(ttnn.to_torch(device_tensor), torch_tensor)
    finally:
        ttnn.close_device(device)
    print(DONE_MARKER, flush=True)


def test_large_file_backed_tensor_upload_finishes():
    GENERATED_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=GENERATED_DIR, prefix="test_load_tensor_pinned_upload_") as file_dir:
        result = subprocess.run(
            [sys.executable, __file__, str(Path(file_dir) / "large_pinned_upload.tensorbin")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=CHILD_DEADLINE_S,
        )
    tail = "\n".join(result.stdout.splitlines()[-40:])
    assert (
        result.returncode != -signal.SIGALRM
    ), f"load_tensor did not finish within {LOAD_DEADLINE_S} s (killed by SIGALRM):\n{tail}"
    assert result.returncode == 0 and DONE_MARKER in result.stdout, f"child exited {result.returncode}:\n{tail}"


if __name__ == "__main__":
    _child(sys.argv[1])
