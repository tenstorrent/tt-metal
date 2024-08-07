ARG IMAGE_TAG=latest

FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-amd64:${IMAGE_TAG}

WORKDIR /usr/app

ARG UID=1000
ARG GID=1000

RUN groupadd -g "${GID}" python \
    && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" python


RUN chown python:python -R /usr/app

USER python

ARG WHEEL_FILENAME
ENV WHEEL_FILENAME=${WHEEL_FILENAME}

COPY $WHEEL_FILENAME /usr/app/

RUN pip3 install $WHEEL_FILENAME 

CMD ["tail", "-f", "/dev/null"]
