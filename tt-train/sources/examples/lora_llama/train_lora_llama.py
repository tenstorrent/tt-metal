# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal single-device Llama fine-tuning with LoRA on Shakespeare."""

import numpy as np
import ttnn
import ttml

from ttml.common.data import (
    CharTokenizer,
    load_shakespeare_text,
    get_batch,
    build_causal_mask,
)
from ttml.common.utils import set_seed
from ttml.models.llama import Llama, LlamaConfig
from ttml.modules import inject_lora

# ── Config ────────────────────────────────────────────────────────────────────

SEQ_LEN = 64
BATCH_SIZE = 4
STEPS = 200
LR = 3e-4
WEIGHT_DECAY = 0.01

LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_linear", "kv_linear", "out_linear"]

# ── Data ──────────────────────────────────────────────────────────────────────

set_seed(42)

text = load_shakespeare_text()
tokenizer = CharTokenizer(text)
vocab_size = (tokenizer.vocab_size + 31) // 32 * 32

ids = np.array(tokenizer.encode(text), dtype=np.uint32)
n_train = int(len(ids) * 0.9)
train_ids = ids[:n_train]

# ── Device ────────────────────────────────────────────────────────────────────

ttml.autograd.AutoContext.get_instance().open_device([1, 1], [0])

# ── Model ─────────────────────────────────────────────────────────────────────

model = Llama(
    LlamaConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab_size,
        max_position_embeddings=SEQ_LEN,
    )
)

model = inject_lora(
    model, rank=LORA_RANK, alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES
)

# ── Train only LoRA parameters ────────────────────────────────────────────────

all_params = model.parameters()
lora_params = {k: v for k, v in all_params.items() if "lora_A" in k or "lora_B" in k}

print(f"Total params: {len(all_params)}")
print(f"Trainable (LoRA): {len(lora_params)}")
for name, tensor in sorted(all_params.items()):
    print(f"  {'*' if name in lora_params else ' '} {name}: {tensor.shape()}")

# ── Optimizer ─────────────────────────────────────────────────────────────────

adamw_cfg = ttml.optimizers.AdamWConfig.make(LR, 0.9, 0.999, 1e-8, WEIGHT_DECAY)
optimizer = ttml.optimizers.AdamW(lora_params, adamw_cfg)

# ── Training loop ─────────────────────────────────────────────────────────────

causal_mask = build_causal_mask(SEQ_LEN)
tt_mask = ttml.autograd.Tensor.from_numpy(
    causal_mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
)

model.train()

for step in range(1, STEPS + 1):
    x_np, y_np = get_batch(train_ids, SEQ_LEN, BATCH_SIZE)

    tt_x = ttml.autograd.Tensor.from_numpy(
        x_np.reshape(BATCH_SIZE, 1, 1, SEQ_LEN),
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.UINT32,
    )
    tt_y = ttml.autograd.Tensor.from_numpy(
        y_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
    )

    optimizer.zero_grad()
    logits = model(tt_x, tt_mask)
    loss = ttml.ops.loss.cross_entropy_loss(logits, tt_y, ttml.ops.ReduceType.MEAN)
    loss.backward(retain_graph=False)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    optimizer.step()

    if step % 10 == 0 or step == 1:
        loss_val = loss.to_numpy(ttnn.DataType.FLOAT32).item()
        print(f"step {step:>4}/{STEPS}  loss={loss_val:.4f}")

ttml.autograd.AutoContext.get_instance().close_device()
