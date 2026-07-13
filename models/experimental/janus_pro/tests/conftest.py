# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--dummy_weights",
        action="store",
        default=False,
        type=bool,
        help="Use dummy/random weights instead of loading checkpoints in tests that support it.",
    )


@pytest.fixture
def dummy_weights(request):
    return request.config.getoption("--dummy_weights") or False


@pytest.fixture(autouse=True)
def _host_tilize_dummy_weights(dummy_weights, monkeypatch):
    """Force host-side tilization (no on-disk cache) for --dummy_weights runs.

    Base ttnn.as_tensor tilizes ON DEVICE when cache_file_name is None (the
    dummy path), which overflows P150 L1 on Janus's 7B QKV and hangs. The
    on-disk cache path avoids that by tilizing on host and only then moving the
    tiled tensor to device -- but persisting it introduces a stale-cache
    footgun: the cache is keyed by tensor name (not weight content), so an
    older run's weights (older code, a different init, a different transformers
    version) load silently and mismatch this run's fresh reference -> spurious
    PCC failures that look like device/precision bugs.

    This patches ttnn.as_tensor to take the host-tilize path WITHOUT writing to
    disk: it mirrors the cache-miss branch of ttnn's own from_torch_and_dump
    (build the tiled tensor on host, then move it to device) minus the dump. So
    dummy runs get fresh, correct weights every time, no device-tilize hang,
    and no cache to go stale. No-op for real-weight runs.
    """
    if not dummy_weights:
        yield
        return

    import ttnn

    def _host_tilize_as_tensor(
        tensor,
        dtype=None,
        *,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=None,
        memory_config=None,
        cache_file_name=None,  # deliberately ignored: never touch disk for dummy weights
        preprocess=None,
        mesh_mapper=None,
    ):
        if device is not None and memory_config is None:
            raise RuntimeError("memory_config must be specified when device is specified")
        if preprocess is not None:
            tensor = preprocess(tensor)
        # Tilize on host (device=None) so the wide 7B weights never live-tilize on
        # device, then move the already-tiled tensor to device. Replicated tensors
        # are built unsharded on host and broadcast by .to(device), matching
        # ttnn.as_tensor's own from_torch_and_dump handling.
        host_tensor = ttnn.from_torch(
            tensor,
            dtype=dtype,
            layout=layout,
            memory_config=memory_config,
            device=None,
            mesh_mapper=None if isinstance(mesh_mapper, ttnn.ReplicateTensorToMeshWrapper) else mesh_mapper,
        )
        if device is not None:
            host_tensor = host_tensor.to(device, memory_config)
        return host_tensor

    monkeypatch.setattr(ttnn, "as_tensor", _host_tilize_as_tensor)
    yield
