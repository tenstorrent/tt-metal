"""Per-step divergence check: run the REAL ONNX estimator and our torch reference
estimator through the SAME 10-step Euler solve independently (each side maintains
its own running x, updated by its own dphi_dt each step, exactly like
CausalConditionalCFMRef.solve_euler), starting from the identical initial noise,
mu, mask, spks, cond. Compare the running state x after every step to see the
SHAPE of the divergence: constant small error (single-component bug), growing
error (compounding through the ODE integration), or a sudden jump at one step
(schedule/timestep-specific bug).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/user/tt-metal")
sys.path.insert(0, "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad")

from cv2_frontend import extract_prompt_feat, extract_speech_tokens, extract_spk_embedding

from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file

print("=== loading real flow.pt + reference audio (same recipe as onnx_cross_check.py) ===")
flow_sd = load_checkpoint_file("flow.pt")

from huggingface_hub import hf_hub_download

campplus_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="campplus.onnx")
st_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="speech_tokenizer_v2.onnx")
onnx_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="flow.decoder.estimator.fp32.onnx")

import onnxruntime as ort

campplus_session = ort.InferenceSession(campplus_path, providers=["CPUExecutionProvider"])
st_session = ort.InferenceSession(st_path, providers=["CPUExecutionProvider"])
estimator_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

import torchaudio
from datasets import load_dataset

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
ref_ex = ds[0]
ref_wav16 = torch.tensor(ref_ex["audio"]["array"], dtype=torch.float32).unsqueeze(0)
ref_wav24 = torchaudio.functional.resample(ref_wav16, 16000, 24000)
prompt_tokens = extract_speech_tokens(st_session, ref_wav16)
prompt_feat = extract_prompt_feat(ref_wav24)
ref_embedding = extract_spk_embedding(campplus_session, ref_wav16)
token_len = min(prompt_feat.shape[1] // 2, prompt_tokens.shape[1])
prompt_feat = prompt_feat[:, : 2 * token_len]
prompt_tokens = prompt_tokens[:, :token_len]

from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef

flow_ref = CausalMaskedDiffWithXvecRef.from_checkpoint(flow_sd)
flow_ref.eval()

torch.manual_seed(123)
new_tokens = torch.randint(0, 6561, (1, 80))

print("=== building the SAME real (mu, mask, spks, cond) as onnx_cross_check.py / attn_cross_check.py ===")
with torch.no_grad():
    spks = torch.nn.functional.normalize(ref_embedding, dim=1)
    spks = flow_ref.spk_embed_affine_layer(spks)

    full_token = torch.cat([prompt_tokens, new_tokens], dim=1)
    tok_emb = flow_ref.input_embedding(full_token.clamp(min=0))
    h = flow_ref.encoder(tok_emb)
    mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
    h = flow_ref.encoder_proj(h)

    conds = torch.zeros(1, mel_len1 + mel_len2, flow_ref.output_size, dtype=h.dtype)
    conds[:, :mel_len1] = prompt_feat

    mu_cl = h  # [1, T, 80] channels-last
    T = mel_len1 + mel_len2
    mask_cl = torch.ones(1, T, 1, dtype=h.dtype)

cfm = flow_ref.decoder  # CausalConditionalCFMRef, real loaded weights
estimator = cfm.estimator

n_timesteps = 10
b = 1
z_cl = cfm.rand_noise[:, :T, :]  # [1, T, 80] channels-last, SAME initial noise both sides
t_span = torch.linspace(0, 1, n_timesteps + 1)
t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)  # cosine scheduler, matches CausalConditionalCFMRef

# ---- two independent running states, identical start ----
x_torch = z_cl.clone()
x_onnx_cf = z_cl.clone().transpose(1, 2).contiguous()  # [1, 80, T] channel-first, ONNX's own convention

mu_onnx_cf = mu_cl.transpose(1, 2).contiguous()
mask_onnx_cf = mask_cl.transpose(1, 2).contiguous()
cond_onnx_cf = conds.transpose(1, 2).contiguous()

t = t_span[0].unsqueeze(0)
dt = t_span[1] - t_span[0]

results = []
print(f"\n=== running {n_timesteps}-step Euler solve, BOTH sides independently, T={T} frames ===")
with torch.no_grad():
    for step in range(1, len(t_span)):
        # --- our torch reference estimator, channels-last ---
        x_in = torch.zeros(2 * b, T, 80, dtype=x_torch.dtype)
        mask_in = torch.zeros(2 * b, T, 1, dtype=x_torch.dtype)
        mu_in = torch.zeros(2 * b, T, 80, dtype=x_torch.dtype)
        t_in = torch.zeros(2 * b, dtype=x_torch.dtype)
        spks_in = torch.zeros(2 * b, spks.shape[-1], dtype=x_torch.dtype)
        cond_in = torch.zeros(2 * b, T, 80, dtype=x_torch.dtype)
        x_in[:] = x_torch
        mask_in[:] = mask_cl
        mu_in[:b] = mu_cl
        t_in[:] = t
        spks_in[:b] = spks
        cond_in[:b] = conds
        dphi_dt = estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
        dphi_dt, cfg_dphi_dt = dphi_dt[:b], dphi_dt[b:]
        dphi_dt_torch = (1.0 + cfm.inference_cfg_rate) * dphi_dt - cfm.inference_cfg_rate * cfg_dphi_dt
        x_torch_next = x_torch + dt * dphi_dt_torch

        # --- real ONNX estimator, channel-first, own independent running state ---
        x_in_cf = np.zeros((2 * b, 80, T), dtype=np.float32)
        mask_in_cf = np.zeros((2 * b, 1, T), dtype=np.float32)
        mu_in_cf = np.zeros((2 * b, 80, T), dtype=np.float32)
        t_in_cf = np.zeros((2 * b,), dtype=np.float32)
        spks_in_cf = np.zeros((2 * b, spks.shape[-1]), dtype=np.float32)
        cond_in_cf = np.zeros((2 * b, 80, T), dtype=np.float32)
        x_in_cf[:] = x_onnx_cf.numpy()
        mask_in_cf[:] = mask_onnx_cf.numpy()
        mu_in_cf[:b] = mu_onnx_cf.numpy()
        t_in_cf[:] = t.numpy()
        spks_in_cf[:b] = spks.numpy()
        cond_in_cf[:b] = cond_onnx_cf.numpy()
        onnx_out = estimator_session.run(
            None,
            {
                "x": x_in_cf,
                "mask": mask_in_cf,
                "mu": mu_in_cf,
                "t": t_in_cf,
                "spks": spks_in_cf,
                "cond": cond_in_cf,
            },
        )[0]
        onnx_out_t = torch.from_numpy(onnx_out)
        dphi_onnx, cfg_dphi_onnx = onnx_out_t[:b], onnx_out_t[b:]
        dphi_dt_onnx = (1.0 + cfm.inference_cfg_rate) * dphi_onnx - cfm.inference_cfg_rate * cfg_dphi_onnx
        x_onnx_cf_next = x_onnx_cf + dt * dphi_dt_onnx

        # --- compare running state AFTER this step's update, same layout ---
        x_torch_next_cf = x_torch_next.transpose(1, 2).contiguous()
        from models.common.utility_functions import comp_pcc

        _, pcc = comp_pcc(x_onnx_cf_next, x_torch_next_cf, 0.99)
        max_diff = (x_onnx_cf_next - x_torch_next_cf).abs().max().item()
        mean_diff = (x_onnx_cf_next - x_torch_next_cf).abs().mean().item()
        x_std = x_onnx_cf_next.std().item()
        results.append((step, float(pcc) if not isinstance(pcc, float) else pcc, max_diff, mean_diff, x_std))
        print(
            f"step {step:2d}: t={t.item():.4f} dt={dt.item():.4f}  PCC={pcc}  "
            f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  onnx_x_std={x_std:.5f}"
        )

        x_torch = x_torch_next
        x_onnx_cf = x_onnx_cf_next
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t

print("\n=== SUMMARY TABLE ===")
print(f"{'step':>4} | {'PCC':>10} | {'max_diff':>10} | {'mean_diff':>10} | {'state_std':>10}")
for step, pcc, max_diff, mean_diff, x_std in results:
    print(f"{step:>4} | {pcc:>10.6f} | {max_diff:>10.5f} | {mean_diff:>10.6f} | {x_std:>10.5f}")
