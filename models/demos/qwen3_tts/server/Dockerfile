# Dedicated Qwen3-TTS API server image (Tenstorrent P300 / Blackhole).
# Boots straight into the OpenAI /v1/audio/speech server.
# tt-metal (TT_METAL_HOME) + HF weights + voice embeddings mount at runtime as volumes.
# Build the tt-metal source from branch yito/qwen3-tts first (see build_ttmetal_tts.sh).
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
SHELL ["/bin/bash", "-c"]

# ffmpeg for mp3/aac/opus encoding + speed (atempo) on /v1/audio/speech
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# server + audio deps INTO the dev image's /opt/venv (system pip would miss it)
RUN source /opt/venv/bin/activate && \
    uv pip install --python /opt/venv/bin/python3 \
      fastapi "uvicorn[standard]" python-multipart soundfile librosa \
      transformers safetensors huggingface_hub

COPY entrypoint.sh /usr/local/bin/qwen3tts-entrypoint.sh
RUN chmod +x /usr/local/bin/qwen3tts-entrypoint.sh

# Bake the tt-home reference voices (jim=GUEST, riata=HOST) so the image is self-contained
# and the compose works with no extra volume (it only mounts the HF cache + hugepages).
COPY voices /models/voices

# P300 single chip is only 1 visible device -> CUSTOM cluster; the shipped p150 ([1,1])
# mesh-graph descriptor is what initializes it (see memory tt-p300-single-chip-run).
ENV ARCH_NAME=blackhole \
    TT_METAL_HOME=/work \
    TT_MESH_GRAPH_DESC_PATH=/work/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    QWEN3TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    QWEN3TTS_VOICES_DIR=/models/qwen3_tts_voices \
    TTS_PORT=8003 \
    TT_LOGGER_LEVEL=ERROR \
    VOICE_JIM_REF_AUDIO=/models/voices/jim_reference.wav \
    VOICE_JIM_REF_TEXT=/models/voices/jim_reference.txt \
    VOICE_RIATA_REF_AUDIO=/models/voices/female_reference_24k.wav \
    VOICE_RIATA_REF_TEXT=/models/voices/female_reference.txt

EXPOSE 8003
ENTRYPOINT ["/usr/local/bin/qwen3tts-entrypoint.sh"]
