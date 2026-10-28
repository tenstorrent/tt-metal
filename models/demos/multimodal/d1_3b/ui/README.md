<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# d1-3B inference UI

A one-page web UI for the [d1-3B decision model](../README.md) on a single Wormhole chip. Type a state (or drop
an image), ask one typed question (Yes/No, Pick one, or Score), and get a confidence bar for **every** option
back — in one forward pass, zero generated tokens.

It is a thin wrapper over the existing `TtD1` API:

```
browser ──HTTP(JSON)──► FastAPI (server.py) ──owns the chip, warm traces──► N150
```

## Start

Use the repo venv (a bare `python3` may have a stale `ttnn`):

```bash
cd $TT_METAL_HOME   # the tt-metal checkout
MESH_DEVICE=N150 ./python_env/bin/python models/demos/multimodal/d1_3b/ui/server.py
```

Startup loads the model and captures the six prefill-bucket traces once (~1–2 min), then logs `d1-3B ready`.

## Open

| From | URL |
|---|---|
| The host itself | `http://localhost:8000` |
| Same network | `http://<host-ip>:8000` |
| Over SSH | `ssh -L 8000:localhost:8000 <host>`, then `http://localhost:8000` |

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `MESH_DEVICE` | — | must be `N150` (or one chip of an `N300`) |
| `D1_UI_HOST` | `0.0.0.0` | bind address |
| `D1_UI_PORT` | `8000` | bind port |

## Endpoints

| Method / path | Purpose |
|---|---|
| `GET /` | the single-page UI |
| `GET /health` | `{"ok": true}` once the model is loaded |
| `POST /ask` | `{state?, image?, questions}` → answers with full per-option probabilities |

`POST /ask` body (image is a `data:` URL and, when present, is used instead of `state`):

```json
{
  "state": "I was charged twice, please refund one.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "Charges and refunds", "technical": "App faults"}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["Can wait", "Today", "Blocking now"]}
  }
}
```

## Notes

- **One process owns the chip.** The server holds the N150 and its warm traces for its whole lifetime; keep it
  running and send requests to it. Requests that touch the device are serialised (batch 1).
- **First image request is slower.** The SigLIP2 image processor loads lazily on the first image; for several
  questions over one image the vision tower runs once and is reused.
- **Stop it** with `Ctrl+C` (or kill the process); the device is released on shutdown.
