ARG TT_METAL_DEV_VERSION=latest
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:${TT_METAL_DEV_VERSION} as base

ARG TT_METAL_COMMIT_SHA
ARG TEST_COMMAND

ENV DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-metal
LABEL org.opencontainers.image.description="Run 6U upstream smoke tests"
LABEL org.opencontainers.image.licenses=MIT

## add user
RUN adduser --uid 1000 --shell /bin/bash user
USER user

RUN git clone https://github.com/tenstorrent/tt-metal --recurse-submodules /home/user/tt-metal

WORKDIR /home/user/tt-metal

RUN git reset --hard ${TT_METAL_COMMIT_SHA} && git submodule update --init --recursive

ENV PYTHONPATH=/home/user/tt-metal
ENV LD_LIBRARY_PATH=/home/user/tt-metal/build/lib
ENV TT_METAL_HOME=/home/user/tt-metal
ENV TEST_COMMAND=${TEST_COMMAND}

RUN ls -hal

# Can delete these to reduce container image size?
COPY _tt-metal/build/ ./build/
COPY _tt-metal/runtime/ ./runtime/
COPY ttnn-*.whl ./

RUN WHEEL_FILENAME=$(ls -1 *.whl) && pip3 install $WHEEL_FILENAME

# TODO: Revisit the addition of TRACY_NO_INVARIANT_CHECK https://github.com/tenstorrent/tt-metal/issues/20161
RUN TRACY_NO_INVARIANT_CHECK=1 TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestQuantaGalaxyControlPlaneInit" --gtest_list_tests
RUN TRACY_NO_INVARIANT_CHECK=1 TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardFixture.*" --gtest_list_tests
RUN TRACY_NO_INVARIANT_CHECK=1 TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardProgramFixture.*" --gtest_list_tests
RUN TRACY_NO_INVARIANT_CHECK=1 TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardBufferFixture.ShardedBufferLarge*ReadWrites" --gtest_list_tests
RUN TRACY_NO_INVARIANT_CHECK=1 python3 -c 'import ttnn'
ENTRYPOINT ["sh", "-c", "$TEST_COMMAND"]
