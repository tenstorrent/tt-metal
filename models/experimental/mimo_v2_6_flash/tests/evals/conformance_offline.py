# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The offline half of the API conformance suite: every case of mm_requests.CASES through vLLM's own chat front end
(request validation, chat parsing, media fetching / decoding, the MiMo chat template, the upstream
MiMoV2OmniMultiModalProcessor) with no engine and no device. For each case it reports the HTTP status the server
would answer (exceptions mapped with vLLM's own create_error_response, the way its exception handlers do) and, on
success, the placeholder count of every multimodal item and the prompt length.

Run with the serving environment's Python (vLLM 0.31), on CPU, without the TT platform plugin:

  VLLM_TARGET_DEVICE=cpu VLLM_PLUGINS= HF_HUB_OFFLINE=1 python conformance_offline.py \\
      --model /path/to/MiMo-V2.6-Flash-MOPD --out offline.json [--extra-path DIR] [--cases id1,id2]

`--extra-path` prepends a directory to sys.path (e.g. a `pip install --target` dir holding torchaudio, to see what
the serving environment would do with it installed).
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


def build_front_end(model_path, max_model_len, limits, media_io_kwargs=None):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.renderers.online_renderer import OnlineRenderer
    from vllm.renderers.registry import renderer_from_config

    kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        hf_overrides={"architectures": ["MiMoV2OmniForCausalLM"]},
        max_model_len=max_model_len,
        limit_mm_per_prompt=limits,
        served_model_name=["m"],
    )
    if media_io_kwargs:
        kwargs["media_io_kwargs"] = media_io_kwargs
    vllm_config = EngineArgs(**kwargs).create_engine_config()
    renderer = renderer_from_config(vllm_config)
    # As serve_rows.sh / the TTI spec launch it: --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice
    return OnlineRenderer(
        vllm_config.model_config,
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
        tool_parser="mimo",
        reasoning_parser="mimo",
    )


def status_of(exc):
    """The HTTP status vLLM 0.31's API server answers for an exception raised while rendering a chat request."""
    import pydantic
    from vllm.entrypoints.serve.exception_handling.error_response import create_error_response

    if isinstance(exc, pydantic.ValidationError):
        return 400  # FastAPI RequestValidationError -> handlers/validation.py -> 400
    return create_error_response(exc).error.code


def run_case(front, case, media, loop):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse

    import mm_requests

    rec = {"id": case.id, "want": case.want, "openrouter": case.openrouter, "note": case.note}
    t0 = time.time()
    try:
        req_body = mm_requests.apply_target(case.build(media), case, "m", "vllm")
    except Exception as exc:  # fixture problem, not a server answer
        rec.update(status=None, error=f"fixture: {type(exc).__name__}: {exc}")
        return rec
    rec["request_bytes"] = len(json.dumps(req_body))
    try:
        request = ChatCompletionRequest.model_validate(req_body)
        out = loop.run_until_complete(front.render_chat(request))
        if isinstance(out, ErrorResponse):
            rec.update(status=out.error.code, error=out.error.message[:300])
        else:
            _, engine_inputs = out
            ei = engine_inputs[0]
            rec["status"] = 200
            rec["prompt_tokens"] = len(ei.get("prompt_token_ids") or [])
            placeholders = {}
            for modality, ranges in (ei.get("mm_placeholders") or {}).items():
                placeholders[modality] = [
                    {
                        "offset": r.offset,
                        "length": r.length,
                        "embeds": int(r.is_embed.sum()) if getattr(r, "is_embed", None) is not None else r.length,
                    }
                    for r in ranges
                ]
            rec["placeholders"] = placeholders
            prompt = ei.get("prompt") or ""
            rec["thinking_disabled"] = prompt.rstrip().endswith("<think></think>")
            rec["prompt_tail"] = prompt[-80:]
    except Exception as exc:  # what the API server's exception handlers would turn into a response
        rec["status"] = status_of(exc)
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        rec["trace_tail"] = traceback.format_exc().strip().splitlines()[-3:]
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def measure_requests(front, path, loop):
    """Prompt and per-item token counts for captured benchmark requests (a mimo_openrouter dry-run log written with
    MM_DRYRUN_FULL=1): what each benchmark item costs our server in prefill tokens."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    rows = []
    with open(path) as f:
        for line in f:
            req = dict(json.loads(line)["request"], model="m")
            out = loop.run_until_complete(front.render_chat(ChatCompletionRequest.model_validate(req)))
            ei = out[1][0]
            items = {
                m: [r.length if getattr(r, "is_embed", None) is None else int(r.is_embed.sum()) for r in rs]
                for m, rs in (ei.get("mm_placeholders") or {}).items()
            }
            rows.append({"prompt_tokens": len(ei["prompt_token_ids"]), "items": items, "bytes": len(line)})

    def stats(values):
        values = sorted(values)
        if not values:
            return None
        pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
        return {
            "n": len(values),
            "mean": round(sum(values) / len(values), 1),
            "p50": pick(0.5),
            "p95": pick(0.95),
            "max": values[-1],
        }

    summary = {
        "requests": len(rows),
        "prompt_tokens": stats([r["prompt_tokens"] for r in rows]),
        "request_bytes": stats([r["bytes"] for r in rows]),
    }
    for modality in sorted({m for r in rows for m in r["items"]}):
        summary[f"{modality}_item_tokens"] = stats([t for r in rows for t in r["items"].get(modality, [])])
        summary[f"{modality}_items_per_request"] = stats([len(r["items"].get(modality, [])) for r in rows])
    return summary


def verdict(rec):
    st = rec.get("status")
    if st is None:
        return "FIXTURE"
    ok_wanted = rec["want"] in ("ok", "ok|4xx")
    err_wanted = rec["want"] in ("4xx", "ok|4xx")
    if st == 200 and ok_wanted:
        return "pass"
    if 400 <= st < 500 and err_wanted:
        return "pass"
    if st >= 500:
        return "GAP-5xx"
    return "GAP"


def summarize(recs):
    lines = [
        "| case | OR form | want | status | verdict | items (tokens) | prompt tokens | error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        items = ", ".join(
            f"{mod}:{'/'.join(str(p['embeds']) for p in ps)}" for mod, ps in (r.get("placeholders") or {}).items()
        )
        err = (r.get("error") or "").replace("|", "\\|").replace("\n", " ")[:140]
        lines.append(
            f"| {r['id']} | {'y' if r['openrouter'] else 'n'} | {r['want']} | {r.get('status')} | {r['verdict']} | "
            f"{items} | {r.get('prompt_tokens', '')} | {err} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--model", default=os.environ.get("MIMO_MODEL_PATH"), required=os.environ.get("MIMO_MODEL_PATH") is None
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="", help="comma-separated case ids (default: all)")
    ap.add_argument("--extra-path", default="", help="directory prepended to sys.path (e.g. torchaudio)")
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--limit-mm", default='{"image": 8, "video": 2, "audio": 4}')
    ap.add_argument("--media-io-kwargs", default="", help='JSON, e.g. {"video": {"num_frames": 64}}')
    ap.add_argument("--assets", default=None, help="media cache dir (default MM_ASSETS_DIR or ~/.cache/mimo_mm_evals)")
    ap.add_argument("--requests", default="", help="instead of the cases: token counts of a full dry-run log")
    args = ap.parse_args()

    if args.extra_path:
        sys.path.insert(0, args.extra_path)
    sys.path.insert(0, HERE)
    import mm_requests

    media = mm_requests.Media(args.assets)
    front = build_front_end(
        args.model, args.max_model_len, json.loads(args.limit_mm), json.loads(args.media_io_kwargs or "null")
    )
    info = front.renderer.get_mm_processor().info
    max_per_item = info.get_mm_max_tokens_per_item(args.max_model_len, json.loads(args.limit_mm))
    print(f"max tokens per item (what vLLM checks against max_num_batched_tokens): {max_per_item}")
    if args.requests:
        loop = asyncio.new_event_loop()
        summary = measure_requests(front, args.requests, loop)
        summary["mm_max_tokens_per_item"] = max_per_item
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
        print(json.dumps(summary, indent=1))
        return
    wanted = set(filter(None, args.cases.split(",")))
    recs = []
    # One loop for every case, as in the server: vLLM's media connector keeps an HTTP session bound to its loop.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for case in mm_requests.CASES:
        if wanted and case.id not in wanted:
            continue
        rec = run_case(front, case, media, loop)
        rec["verdict"] = verdict(rec)
        recs.append(rec)
        print(
            f"{rec['id']:32s} {str(rec.get('status')):5s} {rec['verdict']:8s} {rec.get('placeholders', '')} "
            f"{(rec.get('error') or '')[:120]}",
            flush=True,
        )
    try:
        import torchaudio  # noqa: F401

        has_torchaudio = True
    except ImportError:
        has_torchaudio = False
    import vllm

    result = {
        "vllm": vllm.__version__,
        "mm_max_tokens_per_item": max_per_item,
        "torchaudio": has_torchaudio,
        "max_model_len": args.max_model_len,
        "limit_mm_per_prompt": json.loads(args.limit_mm),
        "media_io_kwargs": json.loads(args.media_io_kwargs or "null"),
        "cases": recs,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, default=str)
    md = summarize(recs)
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
