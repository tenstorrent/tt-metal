# TT-METAL UBUNTU 20.04 AMD64 DOCKERFILE
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV GTEST_VERSION=1.13.0
ENV DOXYGEN_VERSION=1.9.6

# Install build and runtime deps
COPY /scripts/docker/requirements.txt /opt/tt_metal_infra/scripts/docker/requirements.txt
RUN apt-get -y update \
    && xargs -a /opt/tt_metal_infra/scripts/docker/requirements.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install dev deps
COPY /scripts/docker/requirements_dev.txt /opt/tt_metal_infra/scripts/docker/requirements_dev.txt
RUN apt-get -y update \
    && xargs -a /opt/tt_metal_infra/scripts/docker/requirements_dev.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

## Test Related Dependencies
COPY /scripts/docker/install_test_deps.sh /opt/tt_metal_infra/scripts/docker/install_test_deps.sh
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_test_deps.sh ${GTEST_VERSION} ${DOXYGEN_VERSION}

# Copy remaining convenience scripts
COPY /scripts /opt/tt_metal_infra/scripts
COPY build_metal.sh /scripts/build_metal.sh

# Setup Env variables to setup Python Virtualenv - Install TT-Metal Python deps
ENV TT_METAL_INFRA_DIR=/opt/tt_metal_infra
ENV PYTHON_ENV_DIR=${TT_METAL_INFRA_DIR}/tt-metal/python_env
RUN python3 -m venv $PYTHON_ENV_DIR
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"

# Copy requirements from tt-metal folders with requirements.txt docs
COPY /docs/requirements-docs.txt ${TT_METAL_INFRA_DIR}/tt-metal/docs/.
COPY /tt_metal/python_env/* ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.
RUN python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install setuptools wheel

RUN python3 -m pip install -r ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/requirements-dev.txt
RUN python3 -m pip install -r ${TT_METAL_INFRA_DIR}/tt-metal/docs/requirements-docs.txt

CMD ["tail", "-f", "/dev/null"]
