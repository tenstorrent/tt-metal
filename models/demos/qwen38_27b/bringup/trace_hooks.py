# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The upstream HF trace's cache as qwen38_27b's reference state: the trace-state job (T.1) owns this file; no other job
may edit it. Only a run whose spec sets golden.trace (pipeline input goldens_path) uses it.

Plain key/value attention needs nothing here: a layer whose state is [key, value] takes the trace's
past_key_values keys / values [1, kv_heads, N, head_dim] as [kv_heads, seq, head_dim]. For any other cache define

    def trace_state(spec, layer, cache, seq):
        return {state name: tensor}

``cache`` is the layer's cache/layer_XX.safetensors from the trace with the layer prefix taken off the names
(``keys``, ``values``, ``k_pe``, ``self_attn.kv_a_layernorm.output``, ``recurrent_states``, ...; the trace's
metadata.json describes every one). Return exactly the layer's state names (spec.state_names(layer)) in the layout of
the reference's state_tensors(state, layer, seq) as of position seq: a tensor that grows along the sequence cut to its
first seq positions (however it is compressed), a fixed-size one (state.fixed) as it is. Check:
python -m bringup_fw.reference.check_trace --state. This package never imports the framework (bringup_fw).
"""


def trace_state(spec, layer, cache, seq):
    """attn: keys / values [1, kv_heads, N, head_dim] (post k_norm and RoPE, as the reference's KVState) cut to seq;
    linear (Gated DeltaNet): conv_states [1, conv_dim, K] and recurrent_states [1, heads, k_dim, v_dim] as of the
    trace's last position (fixed-size state), batch dim dropped."""
    if "keys" in cache:
        return {"key": cache["keys"][0, :, :seq], "value": cache["values"][0, :, :seq]}
    return {"conv": cache["conv_states"][0], "recurrent": cache["recurrent_states"][0]}
