# Reuse of completed serving image

The stock override supports reuse for TTI task-list-only commits. Supply:

```text
-f docker-image=ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:61962ce48cd4561ed02080d50b7403eea5cb1b2122a160e212ee4ced875c9005
```

Keep explicit `impl-of-model=gemma4-31b-autoport-v3` and the new immutable `inference-server-git-ref`. Completed SWE42200513517 pulled this digest twice from the TT439c/plugin ca7 build tag.

The override skips SHA resolution/build jobs. The test job independently checks out the requested TTI ref, generates its runtime model JSON and binds it into the server; the loader prefers it over the image catalog. Dev mode also mounts reference_config, utils, tests and application src from that checkout. Task-list configuration therefore changes without rebuilding TT. TT source/model adapter, native libraries, plugin and installed vLLM remain in the digest-pinned image.

This applies only while those runtime/dependency assumptions remain unchanged. TT/plugin dispatch ref fields do not validate or rebuild an explicit image. The completed-run image provenance binds them here. No dispatch or pull was performed; saved build time is an estimate until a normal reuse run measures it. See result.json for exact source references and limits.
