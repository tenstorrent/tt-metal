# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""SP ring-joint SDPA over the block-cyclic SP KV cache — **P5 stub, P8 implements it**.

``BRINGUP_RECIPE.md:698-700`` / ``03_OUTLINE.md`` §3.12: P5 creates this file with a
``NotImplementedError`` and a docstring pointing at the template, so that the SP branch in
``prefill.py`` has a real symbol to call and P8 has one place to fill in.

**Template to port:** ``models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:41``
(``dense_sp_attention``), the op call ``:106``, ``persistent_output_buffer_k/v`` ``:116`` / ``:119``,
``joint_strategy="rear"`` ``:122``, ``dim=2`` ``:126``, ``cluster_axis`` ``:129``,
``multi_device_global_semaphore=...ring_attention_ccl_semaphore_handles`` ``:127``,
``ccl_core_grid_offset`` ``:133``, ``use_column_major_ccl=True`` ``:134``, ``is_causal=True``
``:135``, ``kv_cache_batch_idx=slot_idx*num_layers+layer_idx`` ``:141``, ``kv_actual_isl`` ``:142``.

**Deletions when porting** (Llama has neither feature): ``attention_sink=`` (``:144``),
``sliding_window_size=`` (``:145``), and ``_gather_seq_len`` (``:30``) — which for Llama's
full-causal attention collapses to ``return full_seq`` (``:36``), so the persistent buffer key is
simply ``cache_global``.

**Two inherited constraints P8 must not re-litigate:**

* the cache must be ``bfloat8_b`` — asserted by the template at ``:77-81``
  ("KV_CACHE_DTYPE=bf16 is not supported for chunked prefill"). This is what forces ``DEC-017``.
* the ring op requires ``fp32_dest_acc_en=False``
  (``models/demos/gpt_oss_d_p/tt/attention/prefill.py:200``), so this path builds its **own**
  compute-kernel config and must not reuse ``ProgramConfig.get_compute_kernel_config`` (whose
  default is ``True``, ``DEC-031``).
* the SDPA program grid stays **8x8**; ``ring_joint_sdpa_device_operation.cpp:421`` asserts
  ``ccl_core_grid_offset.x >= sdpa_grid.x`` with the offset pinned at ``grid.x - 1 = 11`` on this
  Blackhole (Appendix F.8 / ``DEC-012``). ``ProgramConfig.assert_sdpa_grid_fits`` already enforces
  it at build time.

Also unported by design: ``prefill.py``'s SP one-shot bootstrap (all-gather Q/K/V -> plain SDPA ->
reduce-scatter, ``gpt_oss prefill.py:233-256``) — ``DEC-021`` keeps it, gated off the default.
"""

from __future__ import annotations


def dense_sp_attention(*args, **kwargs):  # noqa: D401 - stub
    """Not implemented in P5. See the module docstring for the exact port.

    Raises:
        NotImplementedError: always. Failing loud is the point: a silent fallback to a plain
            ``is_causal`` SDPA over an SP-sharded Q would compute attention over one chip's
            sequence shard only and still return a correctly-shaped, plausible-looking tensor.
    """
    raise NotImplementedError(
        "llama31_8b_d_p: dense_sp_attention (SP ring-joint SDPA over the block-cyclic KV cache) is "
        "a P8 deliverable. P5 ships the single-card causal SDPA path only. Port "
        "models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:41, dropping attention_sink / "
        "sliding_window_size / _gather_seq_len; keep the cache at bfloat8_b (DEC-017), the SDPA "
        "grid at 8x8 (DEC-012) and fp32_dest_acc_en=False for the ring op (DEC-031)."
    )
