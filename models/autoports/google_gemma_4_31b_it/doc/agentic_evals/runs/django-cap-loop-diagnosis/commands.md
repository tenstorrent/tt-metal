# Prepared commands (coordinator launch only)

Private driver: bringup/artifacts/django-cap-loop/diagnose.py. Syntax checked only; no model loaded or device opened by preparation.

First obtain stock tokenization:

```sh
curl --fail-with-body --max-time 120 -sS -H 'Content-Type: application/json' --data-binary @bringup/artifacts/django-cap-loop/tokenize-request.json http://127.0.0.1:8000/tokenize -o bringup/artifacts/django-cap-loop/tokenize-response.json
```

Inspect returned token array versus case.json. Only after exact array equality set case.metadata.server_tokenization_verified=true. If arrays differ, preserve old hash and replace token_ids with actual stock tokenization, explaining differences. Original runtime had42823 tokens but offline currently42822; neither add_special_tokens setting changes the count. Pinned stock chat protocol default isFalse. No arbitrary token padding or BOS insertion.

HF pilot/full-prefix attempt, hard outer1h bound and internal3500s projection limit:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout --signal=TERM --kill-after=30s 3600s python_env/bin/python bringup/artifacts/django-cap-loop/diagnose.py hf --chunk-size 512 --pilot-tokens 1024 --budget-seconds 3500 --output bringup/artifacts/django-cap-loop/hf512 > bringup/artifacts/django-cap-loop/hf512.log 2>&1
```

Driver loads pinned full60 HF, verifies short one-shot versus cached-chunk distribution (KL<=.01 and same top1), then every actual prefix token in CPU chunks. Short control is only numeric cache validation. It records RSS and each chunk time, projects remaining cost with explicitly heuristic2x linear allowance, stops if projection exceeds budget, and saves only terminal full logits privately. If512 chunk RSS is unsafe, use256 in a separate measured attempt; do not silently truncate prefix. It cannot prove original sampling trajectory identity.

TT selected terminal logits, only after coordinator device-owner gate and server teardown:

```sh
timeout --signal=TERM --kill-after=30s 1800s python_env/bin/python bringup/artifacts/django-cap-loop/diagnose.py tt --batch 8 --output bringup/artifacts/django-cap-loop/tt-b8 > bringup/artifacts/django-cap-loop/tt-b8.log 2>&1
```

TT uses full60/262144context/measured default precision/1x4, existing prefill_logits_continuation over whole prefix with only terminal logit row returned. No tasks commands execute. This is standalone conditional evidence, not the served scheduler/cache state of original turn.

Once both complete:

```sh
python_env/bin/python bringup/artifacts/django-cap-loop/diagnose.py compare --hf bringup/artifacts/django-cap-loop/hf512/logits.pt --tt bringup/artifacts/django-cap-loop/tt-b8/logits.pt --output bringup/artifacts/django-cap-loop/comparison
```

Metrics: full-vocabulary KL(HF||TT), JS, shift-centered maximum logit error, top1 agreement, and release T1/top20/top_p.95 KL with explicit infinite support mismatch. No model-inherent loop verdict from one conditional distribution. Device box remains0.5h total, not a separate1800s allowance per retry.
