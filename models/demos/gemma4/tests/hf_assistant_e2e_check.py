# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-HF ground-truth check for the Gemma4 it-assistant drafter.

Runs the candidate-generator's drafter entirely in HF (real target -> true
post-norm hidden + shared_kv_states -> assistant). Two checks:

  1. First-step: does the assistant predict the target's greedy next token?
  2. Greedy acceptance ceiling: plain-greedy generate with the target while
     recording (token, post-norm hidden, shared_kv, pos) per step; then replay
     the drafter at every step and compare its K drafts to the target's actual
     greedy continuation. Greedy spec-decode commits exactly the target greedy
     trajectory, so this teacher-forced replay equals the true acceptance rate.

    python models/demos/gemma4/tests/hf_assistant_e2e_check.py
"""
import os

import torch
from transformers import AutoTokenizer

TARGET = os.getenv("HF_MODEL", "google/gemma-4-12B-it")
ASSISTANT = os.getenv("GEMMA4_ASSISTANT_MODEL", "google/gemma-4-12B-it-assistant")
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[os.getenv("GEMMA4_SPEC_DTYPE", "fp32")]
PROMPT = os.getenv("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
N_GREEDY = int(os.getenv("GEMMA4_SPEC_CEILING_TOKENS", 64))
DRAFT_LEN = int(os.getenv("GEMMA4_SPEC_DRAFT_LEN", 4))


def main():
    from transformers import Gemma4UnifiedAssistantForCausalLM, Gemma4UnifiedForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(TARGET)
    msgs = [{"role": "user", "content": "The capital of France is"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", tokenize=True)
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    print(f"prompt ids shape={list(ids.shape)} last_tokens={ids[0, -6:].tolist()}")

    print("loading target (fp32, CPU)...")
    full = Gemma4UnifiedForConditionalGeneration.from_pretrained(TARGET, dtype=DTYPE).eval()
    text_model = full.model.language_model
    lm_head = full.lm_head
    get_embed = full.get_input_embeddings()
    print("loading assistant...")
    asst = Gemma4UnifiedAssistantForCausalLM.from_pretrained(ASSISTANT, dtype=DTYPE).eval()

    # DISCRIMINATOR: replay the HF (clean) drafter on TT-exported features vs the
    # TT target greedy chain. Compare to the HF self-ceiling (~1.62) and the TT
    # measured accept (~0.21): result ~0.2 => TT FEATURE precision is the wall;
    # result ~1.6 => residual TT DRAFTER forward bug.
    if os.getenv("GEMMA4_SPEC_TT_FEATURES"):
        replay_tt_features(get_embed, asst)
        return

    with torch.no_grad():
        # Target forward: capture pre-final-norm hidden_states[-1] + shared KV.
        out = text_model(input_ids=ids, use_cache=True, output_hidden_states=True, return_shared_kv_states=True)
        tgt_logits = lm_head(out.last_hidden_state[:, -1:])
        target_first = int(tgt_logits.reshape(-1).argmax())

        hidden_last = out.hidden_states[-1][:, -1:]  # pre-final-norm last-layer hidden
        cur_len = ids.shape[1]
        shared = {k: (v[0][:, :, :cur_len], v[1][:, :, :cur_len]) for k, v in out.shared_kv_states.items()}

        # Assistant first drafter step (exactly SinglePositionMultiTokenCandidateGenerator).
        emb = get_embed(ids[:, -1:])
        inputs_embeds = torch.cat([emb, hidden_last], dim=-1)
        pos = torch.tensor([[cur_len - 1]], dtype=torch.long)
        ao = asst(
            inputs_embeds=inputs_embeds, position_ids=pos, shared_kv_states=shared, attention_mask=None, use_cache=False
        )
        draft_first = int(ao.logits.reshape(-1).argmax())

        # A few autoregressive drafter steps (recurrent hidden).
        drafts = [draft_first]
        last_tok = ao.logits.argmax(dim=-1)
        last_hidden = ao.last_hidden_state
        for _ in range(3):
            emb = get_embed(last_tok)
            inputs_embeds = torch.cat([emb, last_hidden], dim=-1)
            ao = asst(
                inputs_embeds=inputs_embeds,
                position_ids=pos,
                shared_kv_states=shared,
                attention_mask=None,
                use_cache=False,
            )
            last_tok = ao.logits.argmax(dim=-1)
            last_hidden = ao.last_hidden_state
            drafts.append(int(last_tok))

    print(f"\n=== HF ground truth (first step) ===")
    print(f"target greedy first token: {target_first}  ({tok.decode([target_first])!r})")
    print(f"assistant draft0:          {draft_first}  ({tok.decode([draft_first])!r})")
    print(f"assistant drafts[0:4]:     {drafts}")
    print(f"first-token match (drafter==target greedy): {draft_first == target_first}")

    acceptance_ceiling(tok, text_model, lm_head, get_embed, asst)


def _asst_step(asst, get_embed, last_tok, hidden, shared, pos):
    """One drafter step. Returns (next_token_id, next_hidden [1,1,h])."""
    emb = get_embed(last_tok)
    inputs_embeds = torch.cat([emb, hidden], dim=-1)
    ao = asst(
        inputs_embeds=inputs_embeds,
        position_ids=torch.tensor([[pos]], dtype=torch.long),
        shared_kv_states=shared,
        attention_mask=None,
        use_cache=False,
    )
    return ao.logits.argmax(dim=-1), ao.last_hidden_state


def replay_tt_features(get_embed, asst):
    """Replay the clean HF drafter on TT-exported features vs the TT greedy chain.

    Loads the .pt written by test_export_tt_spec_features (per-step TT seed
    hidden + de-paged shared KV + anchor token + pos, and the TT target's greedy
    next token). At each step runs K recurrent drafter steps on the TT features
    and counts the accepted prefix vs the TT greedy continuation. This is the
    same teacher-forced replay as acceptance_ceiling, but with TT (not HF)
    target features + TT (not HF) greedy targets — so it attributes the 0.21 TT
    accept to features (HF drafter also low) vs a TT drafter bug (HF drafter high).
    """
    path = os.environ["GEMMA4_SPEC_TT_FEATURES"]
    data = torch.load(path, weights_only=False)
    steps, greedy, K = data["steps"], data["greedy"], data["draft_len"]
    print(f"\n=== HF drafter on TT features vs TT greedy (K={K}) ===")
    print(f"prompt={data.get('prompt')!r}  steps={len(steps)}  dtype={DTYPE}")
    accepts = []
    with torch.no_grad():
        for t in range(len(steps) - K):
            s = steps[t]
            last_tok = torch.tensor([[s["token"]]], dtype=torch.long)
            hidden = s["hidden"].to(DTYPE)  # post-norm seed [1,1,h]
            shared = {k: (a.to(DTYPE), b.to(DTYPE)) for k, (a, b) in s["shared"].items()}
            drafts = []
            for _ in range(K):
                last_tok, hidden = _asst_step(asst, get_embed, last_tok, hidden, shared, s["pos"])
                drafts.append(int(last_tok))
            cont = greedy[t : t + K]
            m = K
            for i in range(K):
                if drafts[i] != cont[i]:
                    m = i
                    break
            accepts.append(m)
    mean_acc = sum(accepts) / len(accepts) if accepts else 0.0
    print(f"per-position accepts = {accepts}")
    print(f"mean accepted/iter = {mean_acc:.2f} / {K}  over {len(accepts)} positions")
    print(f"=> committed tokens/verify ~= {mean_acc + 1:.2f}")
    print("INTERPRET: ~0.2 => TT FEATURE precision wall;  ~1.6 => residual TT DRAFTER bug")


def acceptance_ceiling(tok, text_model, lm_head, get_embed, asst):
    """Greedy acceptance ceiling via teacher-forced replay over a real generation."""
    msgs = [{"role": "user", "content": PROMPT}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", tokenize=True)
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]

    # Plain greedy generation, recording per-step (token, post-norm hidden, shared_kv, pos).
    steps = []  # each: dict(token, hidden[1,1,h], shared{type:(k,v)}, pos)
    greedy_tokens = []
    with torch.no_grad():
        out = text_model(input_ids=ids, use_cache=True, output_hidden_states=True, return_shared_kv_states=True)
        cache = out.past_key_values
        pos = ids.shape[1] - 1
        last_tok = ids[:, -1:]
        for _ in range(N_GREEDY):
            hidden = out.last_hidden_state[:, -1:]  # post-norm hidden at current last token
            cur_len = pos + 1
            shared = {
                k: (v[0][:, :, :cur_len].clone(), v[1][:, :, :cur_len].clone()) for k, v in out.shared_kv_states.items()
            }
            nxt = int(lm_head(hidden).reshape(-1).argmax())
            steps.append({"token": int(last_tok), "hidden": hidden.clone(), "shared": shared, "pos": pos})
            greedy_tokens.append(nxt)
            # advance target by one (decode)
            last_tok = torch.tensor([[nxt]], dtype=ids.dtype)
            pos += 1
            out = text_model(
                input_ids=last_tok,
                past_key_values=cache,
                use_cache=True,
                position_ids=torch.tensor([[pos]], dtype=torch.long),
                output_hidden_states=True,
                return_shared_kv_states=True,
            )

    # Replay drafter at each step; compare K drafts to the target greedy continuation.
    K = DRAFT_LEN
    round_bf16 = os.getenv("GEMMA4_SPEC_ROUND_BF16") == "1"  # round drafter inputs to bf16 precision
    noise = float(
        os.getenv("GEMMA4_SPEC_NOISE", 0.0)
    )  # relative gaussian noise std on seed+KV (emulate TT target fidelity)

    def _r(x):
        if noise > 0:
            x = x + torch.randn_like(x) * (x.norm() / (x.numel() ** 0.5)) * noise
        if round_bf16:
            x = x.bfloat16().to(x.dtype)
        return x

    def _pcc_log(tag, a, b):
        a, b = a.reshape(-1).float(), b.reshape(-1).float()
        p = 1.0 if torch.allclose(a, b) else float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        return p

    accepts = []
    seed_pccs = []
    with torch.no_grad():
        for t in range(len(steps) - K):
            s = steps[t]
            orig_hidden = s["hidden"]
            s = {**s, "shared": {k: (_r(a), _r(b)) for k, (a, b) in s["shared"].items()}}
            last_tok = torch.tensor([[s["token"]]], dtype=ids.dtype)
            hidden = _r(orig_hidden)
            seed_pccs.append(_pcc_log("seed", hidden, orig_hidden))
            drafts = []
            for _ in range(K):
                last_tok, hidden = _asst_step(asst, get_embed, last_tok, hidden, s["shared"], s["pos"])
                hidden = _r(hidden)  # TT drafter recurrent hidden is bf16
                drafts.append(int(last_tok))
            target_cont = greedy_tokens[t : t + K]
            m = K
            for i in range(K):
                if drafts[i] != target_cont[i]:
                    m = i
                    break
            accepts.append(m)

    mean_acc = sum(accepts) / len(accepts) if accepts else 0.0
    print(f"\n=== HF greedy acceptance ceiling (K={K}) ===")
    print(f"prompt={PROMPT!r}")
    print(f"greedy text: {tok.decode(greedy_tokens)!r}")
    mean_seed_pcc = sum(seed_pccs) / len(seed_pccs) if seed_pccs else 1.0
    print(f"input noise={noise} round_bf16={round_bf16}  mean seed PCC(perturbed vs orig)={mean_seed_pcc:.4f}")
    print(f"mean accepted/iter = {mean_acc:.2f} / {K}  over {len(accepts)} positions")
    print(f"per-position accepts = {accepts}")
    # Expected committed tokens per verify = mean_acc + 1; theoretical speedup ~ that / 1.
    print(f"=> committed tokens/verify ~= {mean_acc + 1:.2f}  (ideal greedy speedup ceiling)")


if __name__ == "__main__":
    main()
