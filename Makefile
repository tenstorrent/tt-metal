.SUFFIXES:

MAKEFLAGS := --jobs=$(shell nproc)

# Setup CONFIG, DEVICE_RUNNER, and out/build dirs first
TT_METAL_HOME ?= $(shell git rev-parse --show-toplevel)
ARCH_NAME ?= grayskull

include ./module.mk
