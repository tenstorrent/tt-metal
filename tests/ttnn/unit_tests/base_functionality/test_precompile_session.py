# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _eltwise_chain(device, torch_a, torch_b):
    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.mul(ttnn.add(ttnn.gelu(a), b), b)


def _collect(session, fn):
    session.begin_collect(real_alloc=True)
    try:
        fn()
    finally:
        session.end_collect()


def test_collected_kernels_build_and_real_run_matches(device):
    torch.manual_seed(0)
    torch_a = torch.rand((1, 1, 64, 128), dtype=torch.bfloat16)
    torch_b = torch.rand((1, 1, 64, 128), dtype=torch.bfloat16)
    expected = (torch.nn.functional.gelu(torch_a.float()) + torch_b.float()) * torch_b.float()

    session = ttnn.graph.CompileSession(workers=4)
    _collect(session, lambda: _eltwise_chain(device, torch_a, torch_b))
    session.wait()
    stats = session.stats()
    session.close()

    assert stats.programs_added == 3
    assert stats.kernels_queued > 0
    assert stats.kernels_failed == 0
    assert stats.kernels_built + stats.kernels_preempted == stats.kernels_queued

    output = ttnn.to_torch(_eltwise_chain(device, torch_a, torch_b))
    assert_with_pcc(expected, output.float(), 0.99)


def test_repeated_program_is_added_once(device):
    torch_a = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)

    session = ttnn.graph.CompileSession(workers=2)

    def body():
        a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn.exp(a)
        ttnn.exp(a)

    _collect(session, body)
    session.close()
    stats = session.stats()
    assert stats.programs_added == 1
    assert stats.programs_deduped == 1


def test_spec_factory_program_is_queued_not_compiled(device):
    # permute builds its workload through a Metal 2.0 spec factory, which compiles eagerly unless the
    # session defers it. Had it compiled during the collect, the workers would find its kernels already
    # built and count them as preempted.
    # A shape no other test here uses, so its kernels are not already built in this process.
    torch_x = torch.rand((1, 3, 32, 96), dtype=torch.bfloat16)

    session = ttnn.graph.CompileSession(workers=2)

    def body():
        x = ttnn.from_torch(torch_x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        ttnn.permute(x, (0, 2, 1, 3))

    _collect(session, body)
    session.wait()
    stats = session.stats()
    session.close()
    assert stats.programs_added == 1
    assert stats.kernels_queued > 0
    assert stats.kernels_failed == 0
    assert stats.kernels_preempted == 0
    assert stats.kernels_built == stats.kernels_queued

    x = ttnn.from_torch(torch_x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.to_torch(ttnn.permute(x, (0, 2, 1, 3)))
    assert torch.equal(output, torch_x.permute(0, 2, 1, 3))


def test_builds_survive_device_close_and_reopen():
    # Workers keep building after the device that queued the kernels is closed; reopening the device
    # must not invalidate the build state they use.
    device = ttnn.open_device(device_id=0)
    session = ttnn.graph.CompileSession(workers=2)
    torch_a = torch.rand((1, 1, 64, 64), dtype=torch.bfloat16)
    try:

        def body():
            a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
            ttnn.sigmoid(ttnn.relu(ttnn.neg(a)))

        _collect(session, body)
    finally:
        ttnn.close_device(device)

    device = ttnn.open_device(device_id=0)
    try:
        session.wait()
        stats = session.stats()
        assert stats.kernels_queued > 0
        assert stats.kernels_failed == 0
        a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
        output = ttnn.to_torch(ttnn.sigmoid(ttnn.relu(ttnn.neg(a))))
        assert_with_pcc(torch.sigmoid(torch.relu(-torch_a.float())), output.float(), 0.99)
    finally:
        session.close()
        ttnn.close_device(device)


def test_disabled_program_cache_does_not_dedup(device):
    # With the program cache off every program reports hash 0; nothing may be skipped as a duplicate.
    device.disable_and_clear_program_cache()
    torch_a = torch.rand((1, 1, 32, 64), dtype=torch.bfloat16)
    session = ttnn.graph.CompileSession(workers=2)

    def body():
        a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn.exp(a)
        ttnn.log(a)

    try:
        _collect(session, body)
        session.wait()
        stats = session.stats()
    finally:
        session.close()
        device.enable_program_cache()
    assert stats.programs_deduped == 0
    assert stats.programs_added == 2
    assert stats.kernels_failed == 0


def test_close_drops_queued_kernels(device):
    torch_a = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
    session = ttnn.graph.CompileSession(workers=1)

    def body():
        a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
        for op in (ttnn.cos, ttnn.sin, ttnn.tan, ttnn.tanh, ttnn.erf, ttnn.erfc):
            op(a)

    _collect(session, body)
    session.close()
    stats = session.stats()
    assert stats.kernels_running == 0 and stats.kernels_pending == 0
    accounted = stats.kernels_built + stats.kernels_preempted + stats.kernels_failed + stats.kernels_dropped
    assert accounted == stats.kernels_queued


def test_builds_survive_reopen_with_another_configuration():
    # Kernels queued for one device configuration keep building against it after the device is reopened with
    # another one, and the real run on the first configuration then finds them.
    torch_a = torch.rand((1, 1, 64, 96), dtype=torch.bfloat16)

    def body(device):
        a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn.softplus(ttnn.abs(a))

    session = ttnn.graph.CompileSession(workers=1)
    device = ttnn.open_device(device_id=0, num_command_queues=2)
    try:
        _collect(session, lambda: body(device))
    finally:
        ttnn.close_device(device)
    device = ttnn.open_device(device_id=0)
    try:
        session.wait()
        stats = session.stats()
        assert stats.kernels_queued > 0
        assert stats.kernels_failed == 0
    finally:
        ttnn.close_device(device)
    session.close()

    device = ttnn.open_device(device_id=0, num_command_queues=2)
    try:
        output = ttnn.to_torch(
            ttnn.softplus(ttnn.abs(ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)))
        )
        assert_with_pcc(torch.nn.functional.softplus(torch_a.float().abs()), output.float(), 0.99)
    finally:
        ttnn.close_device(device)
