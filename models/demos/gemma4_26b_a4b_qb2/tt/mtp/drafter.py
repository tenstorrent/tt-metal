# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The it-assistant drafter on the port's TP4 mesh.

Reuses ``models/demos/gemma4/tt/assistant/model.py`` (``Gemma4AssistantModel``: 4 KV-shared decoder layers,
hidden 1024) built from the same ``Gemma4DecoderLayer`` / ``MeshConfig(decode tp=4)`` / CCL manager the port's
layers use, so its Q heads per chip line up with the port's KV-cache heads per chip (sliding: 4 q / 2 kv heads;
full: 4 q / 1 kv head, the GQA head of that chip's q heads). The drafter cross-attends the caches of the target's
last sliding (28) and last full (29) layer.
"""

import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.assistant.model import Gemma4AssistantModel
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.model_config import Gemma4AssistantArgs

DRAFTER_ID = "google/gemma-4-26B-A4B-it-assistant"
DRAFTER_REVISION = "6e5aaaf4c42b98394530b8fda2e95cadd65c151c"


class _TargetView:
    """What ``Gemma4AssistantModel`` reads from its target: decode RoPE tables and the scaled embedding."""

    def __init__(self, model):
        self.model = model
        self.rope_caches_2d = model.rope_decode
        self.hidden_size = model.config.hidden_size

    def embed_tokens(self, tokens):
        return self.model.embed(tokens)


def load_drafter(model, *, max_seq_len=None):
    """Build the drafter next to a loaded ``Gemma4Model`` (full model: needs layers 28 and 29)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(DRAFTER_ID, revision=DRAFTER_REVISION, local_files_only=True)
    args = Gemma4AssistantArgs.from_hf_config(Gemma4AssistantArgs.load_hf_config(path))
    if args.backbone_hidden_size != model.config.hidden_size:
        raise ValueError("Drafter backbone width differs from the target hidden size")
    if args.text_args.num_key_value_heads != model.config.num_key_value_heads:
        raise ValueError("Drafter KV heads differ from the target's")
    max_seq_len = max_seq_len or model.max_seq_len
    args.cluster_shape = tuple(model.mesh.shape)
    args.text_args.max_seq_len = int(max_seq_len)
    state = Gemma4AssistantArgs.load_state_dict(path)
    drafter = Gemma4AssistantModel(
        mesh_device=model.mesh,
        assistant_args=args,
        target_model=_TargetView(model),
        state_dict=state,
        # The demo layers need the gemma4 CCL manager (the port's layers use the gpt_oss one).
        ccl_manager=CCLManager(model.mesh, num_links=1, topology=ttnn.Topology.Linear),
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=MeshConfig(model.mesh.shape, decode=ModeConfig(tp=4)),
        max_local_batch_size=1,
        bounded_sliding_kv_cache=False,
    )
    del state
    return drafter


def shared_layers(model):
    """Indices (in ``model.layers``) of the target's last sliding and last full layer, by layer type."""
    found = {}
    for number, index in enumerate(model.layer_indices):
        found[model.config.layer_types[index]] = number
    if set(found) != {"sliding_attention", "full_attention"}:
        raise ValueError("The drafter needs one sliding and one full target layer")
    return found


def host_ids(tensor, rows):
    """First ``rows`` uint32 ids of a replicated device tensor (device 0)."""
    value = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    return [int(v) for v in value.reshape(-1)[:rows].tolist()]


def upload_ids(mesh, ids, width=None):
    value = torch.zeros(1, width or len(ids), dtype=torch.int64)
    value[0, : len(ids)] = torch.tensor(ids)
    return ttnn.from_torch(
        value,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
