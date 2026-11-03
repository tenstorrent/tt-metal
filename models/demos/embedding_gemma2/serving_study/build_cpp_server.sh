#!/bin/bash
# Build tt-media-server's cpp_server from Gio's TTIS branch (emb comparison), niced, clang-20 toolchain from the emb tt-metal tree.
set -e
E=${EMB_ROOT:-/mnt/tt-data/nkira/muse-glimmer-30b/emb}; TTIS=${EMB_TTIS:-$E/tt-inference-server}
cd $TTIS/tt-media-server/cpp_server
# tokenizers-cpp needs rustc >= 1.80; the host's apt rustc is 1.75. Use a private rustup toolchain under $E.
export RUSTUP_HOME=$E/rustup CARGO_HOME=$E/cargo PATH=$E/cargo/bin:$PATH
# CMake caches CARGO_EXECUTABLE at first configure (/usr/bin/cargo 1.75); cargo 1.75 driving rustc 1.99 gives E0786
# "invalid metadata version" on proc-macro crates. Pin the private cargo and drop stale tokenizers_cpp output.
[ -f build/CMakeCache.txt ] && sed -i "s|^CARGO_EXECUTABLE:FILEPATH=.*|CARGO_EXECUTABLE:FILEPATH=$E/cargo/bin/cargo|" build/CMakeCache.txt
grep -q "/usr/bin/cargo" build/build.ninja 2>/dev/null && rm -rf build/_deps/tokenizers_cpp-build build/build.ninja
echo "start $(git -C $TTIS rev-parse --short HEAD) $(date -u +%FT%TZ)"
# fetch_tokenizers.sh skips gated models when no token resolves, but falls back to ~/.cache/huggingface/token (the
# shared account's, no Llama access) and then fails. Embedding mode never tokenizes in C++, so skip gated models:
# empty token + a scratch HOME without that file.
HOME=$E/build_home HF_TOKEN= HUGGING_FACE_HUB_TOKEN= TT_METAL_HOME=${EMB_TT_METAL:-$E/tt-metal} nice -n 19 ./build.sh --toolchain-path ${EMB_TT_METAL:-$E/tt-metal}/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake
ls -la build/tt_media_server_cpp && echo "build done $(date -u +%FT%TZ)"
