# Serving tests (run against a live server)

Plain Python 3 scripts (standard library; `Pillow` for the image ones) that talk to `http://127.0.0.1:8000`. They were run against the image of this release at 16 users / 262,144 tokens and at 4 users / long context; the results are in `RELEASE_NOTES.md` and `GETTING_STARTED.md`.

| Script | What it checks |
|---|---|
| `sanity.py` | sampled request, streaming, 8 concurrent arithmetic questions |
| `long.py` | a ~20,000-token prompt with a number to retrieve |
| `needle_probe.py` | short questions, 4 concurrent GSM8K 8-shot requests (set `GSM8K_PROMPTS` to the prompt file), then a 100,000-word haystack with the secret code at the end and at the start (the answer must be 4817) |
| `serving_api_test.py <max_model_len>` | 23 OpenAI-API checks: models, chat, completions, streaming + usage, stop, finish reasons, seeds, top-k, the logprobs / `n` policy, too-long prompts, unknown model, malformed bodies, concurrent mixed streaming, client disconnect |
| `overload_test.py <slots>` | 4 x `max_num_seqs` requests at once, two waves: every answer correct, `/health` 200 afterwards |
| `client_img.py` | text only, one image, two images, 4 image + 4 text at once, streaming |
| `client_many.py` | 16 images in one prompt, one maximum-size image, 17 images (must be refused with an OpenAI-style 400) |
| `client_stress.py` | 16 concurrent image requests, then 48 random images (number and colours) |

The GSM8K 8-shot accuracy and the speed sweeps use tt-inference-server's `evals` and `benchmarks` workflows (see the guide).
