# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Ground the reference chain: build real input, run generate(), reproduce the
manual explicit chain, confirm they match, save golden for the TT e2e test."""
import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

torch.manual_seed(0)
MODEL = "Qwen/Qwen2-VL-7B-Instruct"
N = 24
HERE = os.path.dirname(os.path.abspath(__file__))

proc = AutoProcessor.from_pretrained(MODEL)
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()

# small deterministic image -> few vision tokens
img = Image.new("RGB", (112, 112))
px = img.load()
for y in range(112):
    for x in range(112):
        px[x, y] = ((x * 2) % 256, (y * 2) % 256, ((x + y)) % 256)

messages = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the colors in this image."}]}
]
text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = proc(text=[text], images=[img], return_tensors="pt")
print(
    "input_ids",
    inputs.input_ids.shape,
    "pixel_values",
    inputs.pixel_values.shape,
    "grid_thw",
    inputs.image_grid_thw.tolist(),
)
print("num image tokens in prompt:", int((inputs.input_ids == model.config.image_token_id).sum()))

with torch.no_grad():
    gen = model.generate(**inputs, max_new_tokens=N, do_sample=False, num_beams=1)
new_tokens = gen[0, inputs.input_ids.shape[1] :]
print("GEN new tokens:", new_tokens.tolist())
print("GEN text:", proc.batch_decode(new_tokens.unsqueeze(0), skip_special_tokens=True)[0])

# ---- manual explicit chain (the reference chain the TT pipeline reproduces) ----
mdl = model.model  # Qwen2VLModel
with torch.no_grad():
    image_embeds = mdl.get_image_features(inputs.pixel_values.to(model.dtype), inputs.image_grid_thw).pooler_output
    image_embeds = torch.cat(image_embeds, dim=0)
    print("image_embeds", image_embeds.shape)

    cur_ids = inputs.input_ids.clone()
    man_tokens, man_logits = [], []
    for step in range(N):
        emb = mdl.get_input_embeddings()(cur_ids)
        mask = (cur_ids == model.config.image_token_id).unsqueeze(-1).expand_as(emb)
        emb = emb.masked_scatter(mask, image_embeds.to(emb.dtype))
        mm_tt = (cur_ids == model.config.image_token_id).long()
        pos, _ = mdl.get_rope_index(
            cur_ids, mm_tt, image_grid_thw=inputs.image_grid_thw, attention_mask=torch.ones_like(cur_ids)
        )
        out = mdl.language_model(input_ids=None, inputs_embeds=emb, position_ids=pos, use_cache=False)
        logits = model.lm_head(out.last_hidden_state[:, -1, :])
        man_logits.append(logits)
        nt = int(logits.argmax(-1))
        man_tokens.append(nt)
        cur_ids = torch.cat([cur_ids, torch.tensor([[nt]])], dim=1)
    print("MAN tokens:", man_tokens)
    print("MATCH generate == manual:", man_tokens == new_tokens.tolist())

golden = {
    "input_ids": inputs.input_ids,
    "attention_mask": inputs.attention_mask,
    "pixel_values": inputs.pixel_values,
    "image_grid_thw": inputs.image_grid_thw,
    "gen_tokens": new_tokens,
    "man_tokens": torch.tensor(man_tokens),
    "man_logits": torch.stack(man_logits, dim=0).squeeze(1),  # (N, vocab)
    "image_embeds": image_embeds,
    "N": N,
}
torch.save(golden, os.path.join(HERE, "e2e_golden.pt"))
print("saved golden ->", os.path.join(HERE, "e2e_golden.pt"))
