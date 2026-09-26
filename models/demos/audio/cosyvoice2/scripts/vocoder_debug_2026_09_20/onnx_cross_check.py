"""Cross-check our torch reference CFM estimator against the REAL, officially
exported ONNX estimator (flow.decoder.estimator.fp32.onnx) -- independent of
our own hand-ported reference code entirely. If our reference agrees with
this, the estimator itself is confirmed faithful to the real trained model,
not just internally self-consistent between our port and our reference.
"""
import sys

import numpy as np
import torch
import torchaudio

sys.path.insert(0, "/home/user/tt-metal")
sys.path.insert(0, "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad")

from cv2_frontend import extract_prompt_feat, extract_speech_tokens, extract_spk_embedding

from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file

print("=== loading real flow.pt + reference audio ===")
flow_sd = load_checkpoint_file("flow.pt")

from huggingface_hub import hf_hub_download

campplus_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="campplus.onnx")
st_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="speech_tokenizer_v2.onnx")
onnx_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="flow.decoder.estimator.fp32.onnx")

import onnxruntime as ort

campplus_session = ort.InferenceSession(campplus_path, providers=["CPUExecutionProvider"])
st_session = ort.InferenceSession(st_path, providers=["CPUExecutionProvider"])
estimator_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

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

print("=== building the SAME real (x, mask, mu, t, spks, cond) our decoder solves with ===")
with torch.no_grad():
    # Replicate CausalMaskedDiffWithXvecRef.inference's setup exactly, up to
    # (but not including) the CFM solve, so we get a genuine mid-solve
    # (x, mask, mu, t, spks, cond) tuple to feed BOTH the real ONNX estimator
    # and our own torch reference estimator directly.
    assert new_tokens.shape[0] == 1
    spks = torch.nn.functional.normalize(ref_embedding, dim=1)
    spks = flow_ref.spk_embed_affine_layer(spks)

    full_token = torch.cat([prompt_tokens, new_tokens], dim=1)
    tok_emb = flow_ref.input_embedding(full_token.clamp(min=0))
    h = flow_ref.encoder(tok_emb)
    mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
    h = flow_ref.encoder_proj(h)

    conds = torch.zeros(1, mel_len1 + mel_len2, flow_ref.output_size, dtype=h.dtype)
    conds[:, :mel_len1] = prompt_feat
    conds = conds.transpose(1, 2)  # [1, 80, T]

    mu = h.transpose(1, 2).contiguous()  # [1, 80, T]
    mask = torch.ones(1, 1, mel_len1 + mel_len2, dtype=h.dtype)

    cfm = flow_ref.decoder
    torch.manual_seed(0)
    T = mel_len1 + mel_len2
    z = cfm.rand_noise[:, :T, :].transpose(1, 2)  # [1, 80, T] -- matches CausalConditionalCFMRef's own channels-last->channel-first
    t_span = torch.linspace(0, 1, 10 + 1)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t0 = t_span[0].unsqueeze(0)

    b = 1
    x_in = torch.zeros(2 * b, 80, T)
    mask_in = torch.zeros(2 * b, 1, T)
    mu_in = torch.zeros(2 * b, 80, T)
    t_in = torch.zeros(2 * b)
    spks_in = torch.zeros(2 * b, 80)
    cond_in = torch.zeros(2 * b, 80, T)
    x_in[:] = z
    mask_in[:] = mask
    mu_in[:b] = mu
    t_in[:] = t0
    spks_in[:b] = spks
    cond_in[:b] = conds

print(f"x {x_in.shape}, mask {mask_in.shape}, mu {mu_in.shape}, t {t_in.shape}, spks {spks_in.shape}, cond {cond_in.shape}")
print("(channel-first [B, 80, T] -- real upstream's / the ONNX model's own convention)")

# CausalConditionalDecoderRef.forward's OWN docstring: "x/mu/cond: [B, T, 80]"
# (channels-last, this port's convention throughout) and "mask: [B, T, 1]" --
# different from the channel-first tensors above. Transpose before calling our
# estimator, or the concat-along-last-dim inside forward() would silently
# concatenate along the wrong axis instead of erroring.
x_in_cl = x_in.transpose(1, 2).contiguous()
mask_in_cl = mask_in.transpose(1, 2).contiguous()
mu_in_cl = mu_in.transpose(1, 2).contiguous()
cond_in_cl = cond_in.transpose(1, 2).contiguous()

print("=== our torch reference estimator's raw output (one Euler step, step 1) ===")
estimator = flow_ref.decoder.estimator  # CausalConditionalDecoderRef
with torch.no_grad():
    ours_cl = estimator(x_in_cl, mask_in_cl, mu_in_cl, t_in, spks_in, cond_in_cl)
ours = ours_cl.transpose(1, 2).contiguous()  # back to [2, 80, T] for comparison against the ONNX output
print(f"ours: shape={ours.shape} mean={ours.mean().item():.5f} std={ours.std().item():.5f}")

print("=== real official ONNX estimator's raw output, SAME inputs ===")
onnx_out = estimator_session.run(
    None,
    {
        "x": x_in.numpy().astype(np.float32),
        "mask": mask_in.numpy().astype(np.float32),
        "mu": mu_in.numpy().astype(np.float32),
        "t": t_in.numpy().astype(np.float32),
        "spks": spks_in.numpy().astype(np.float32),
        "cond": cond_in.numpy().astype(np.float32),
    },
)[0]
onnx_out_t = torch.from_numpy(onnx_out)
print(f"onnx: shape={onnx_out_t.shape} mean={onnx_out_t.mean().item():.5f} std={onnx_out_t.std().item():.5f}")

print("\n=== COMPARISON ===")
from models.common.utility_functions import comp_pcc

passed, pcc = comp_pcc(onnx_out_t, ours, 0.99)
max_diff = (onnx_out_t - ours).abs().max().item()
rel_diff = ((onnx_out_t - ours).abs() / (onnx_out_t.abs() + 1e-6)).mean().item()
print(f"PCC (our torch ref vs. real official ONNX): {pcc}")
print(f"max abs diff: {max_diff:.6f}")
print(f"mean relative diff: {rel_diff:.6f}")
print(f"PASS (our reference matches the real official model)" if passed else "MISMATCH -- our reference diverges from the real official model")
