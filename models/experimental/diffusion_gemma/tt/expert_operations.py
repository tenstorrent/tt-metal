# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma-local expert activation helpers."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock

import ttnn
import models.demos.gemma4.tt.experts.decode as gemma4_expert_decode
import models.demos.gemma4.tt.experts.prefill as gemma4_expert_prefill


_tanh_expert_active: ContextVar[bool] = ContextVar("diffusion_gemma_tanh_expert", default=False)
_dispatcher_lock = Lock()


def apply_gelu(value):
    """Apply the checkpoint's GELU variant (``hidden_activation: gelu_pytorch_tanh``)."""
    return ttnn.gelu(value, variant=ttnn.GeluVariant.Tanh)


def apply_geglu(gate, up):
    """Apply the configured GeGLU and release the temporary."""
    activated = apply_gelu(gate)
    result = ttnn.mul(activated, up)
    activated.deallocate(True)
    return result


def shared_mlp_forward(mlp, hidden_states):
    """Run a shared MLP with the DiffusionGemma activation contract."""
    gate = ttnn.linear(hidden_states, mlp.gate_proj)
    gate = apply_gelu(gate)
    up = ttnn.linear(hidden_states, mlp.up_proj)
    activated = ttnn.mul(gate, up)
    gate.deallocate(True)
    up.deallocate(True)
    output = ttnn.linear(activated, mlp.down_proj)
    activated.deallocate(True)
    if mlp.mesh_config is not None and mlp.mesh_config.tp > 1:
        from models.experimental.diffusion_gemma.tt.ccl import ccl_allreduce

        output = ccl_allreduce(output, mlp.mesh_config, mlp.ccl_manager)
    return output


def _legacy_geglu_with_release(gate, up):
    """Run gemma4's own GeGLU math, but release the activation temporary here.

    Reproduces ``models.demos.gemma4.tt.experts.operations.apply_geglu`` exactly --
    ``fast_and_approximate_mode`` GELU, *not* this checkpoint's tanh variant -- and adds the
    ``deallocate`` that the shared copy does not do. Keeping the release DG-local is what lets
    ``models/demos/gemma4`` stay byte-identical to ``main``.

    Only reachable when the tanh contextvar is unset, which the live path never does
    (``prefill_moe.py`` pins ``tanh_gelu = True``); it exists so that a future caller which
    forgets the context still cannot leak.
    """
    activated = ttnn.gelu(gate, fast_and_approximate_mode=True)
    result = ttnn.mul(activated, up)
    activated.deallocate(True)
    return result


def _contextual_geglu(gate, up):
    if _tanh_expert_active.get():
        return apply_geglu(gate, up)
    return _legacy_geglu_with_release(gate, up)


def _install_contextual_geglu() -> None:
    if gemma4_expert_decode.apply_geglu is _contextual_geglu and gemma4_expert_prefill.apply_geglu is _contextual_geglu:
        return
    with _dispatcher_lock:
        gemma4_expert_decode.apply_geglu = _contextual_geglu
        gemma4_expert_prefill.apply_geglu = _contextual_geglu


@contextmanager
def use_tanh_expert_activations(enabled: bool = True):
    """Select DG tanh-GELU for shared expert fallbacks in this call context.

    ``enabled`` exists because ``prefill_moe`` threads a computed value through it.
    """
    if not enabled:
        yield
        return
    _install_contextual_geglu()
    token = _tanh_expert_active.set(True)
    try:
        yield
    finally:
        _tanh_expert_active.reset(token)
