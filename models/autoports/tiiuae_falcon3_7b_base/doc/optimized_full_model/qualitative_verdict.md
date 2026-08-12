# Qualitative verdict

The exact `tiiuae/Falcon3-7B-Base` revision `bf3d7ed586cb22a921520e2d681a9d3d7642cde8` tokenizer has `chat_template = null`. Following the qualitative-check workflow, this base checkpoint was evaluated with a plain completion prompt, not an invented chat wrapper. The source is `models/common/readiness_check/autoregressive_prompt.txt`; HF and TT used the same 59 prompt tokens, greedy decoding, and 100-token budget.

Both outputs are coherent English story continuations. HF continues with an unusual sunrise/set premise; TT continues with sunlight/refraction. TT has no wrong language, control leakage, doubled subwords, prompt echo, corrupt first token, mechanical repetition, or cross-request contamination. The first four generated IDs match HF before expected precision-policy divergence. Verdict: pass. See `results/autoregressive/autoregressive_meta.json`, `hf_completion.txt`, and `tt_completion.txt`.
