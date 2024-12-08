ARG BASE_IMAGE_NAME=tt-metalium/ubuntu-20.04-amd64
#
# Currently the release image uses the base image which is also the build image.
# However, in the future, we could point a true base image that is a base for both releases and builds.
# This work is described in https://github.com/tenstorrent/tt-metal/issues/11974
FROM ghcr.io/tenstorrent/tt-metal/$BASE_IMAGE_NAME

ARG WHEEL_FILENAME
ADD $WHEEL_FILENAME $WHEEL_FILENAME
RUN pip3 install $WHEEL_FILENAME
