ARG IMAGE_TAG=latest

FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-amd64:${IMAGE_TAG}

WORKDIR /usr/app

ARG UID=1000
ARG GID=1000

RUN groupadd -g "${GID}" python \
    && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" python


RUN chown python:python -R /usr/app

USER python

ARG ARCH_NAME=wormhole_b0
ARG ARCH_NAME_IN_URL=wormhole.b0
ARG VERSION=0.51.0-rc25

ENV ARCH_NAME=${ARCH_NAME}
ENV VERSION=${VERSION}
RUN VERSION_NO_DASH=$(echo "$VERSION" | sed 's/-//g') && wget https://github.com/tenstorrent/tt-metal/releases/download/v${VERSION}/metal_libs-${VERSION_NO_DASH}+${ARCH_NAME_IN_URL}-cp38-cp38-linux_x86_64.whl
RUN VERSION_NO_DASH=$(echo "$VERSION" | sed 's/-//g') && pip3 install metal_libs-${VERSION_NO_DASH}+${ARCH_NAME_IN_URL}-cp38-cp38-linux_x86_64.whl

#RUN chown python:python -R /usr/local/lib/python3.8/dist-packages/

CMD ["tail", "-f", "/dev/null"]
