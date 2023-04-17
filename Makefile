.SUFFIXES:

MAKEFLAGS := --jobs=8
# nproc can result in OOM errors for specific machines and should be reworked.
#MAKEFLAGS := --jobs=$(shell nproc)

# Setup CONFIG, DEVICE_RUNNER, and out/build dirs first
TT_METAL_HOME ?= $(shell git rev-parse --show-toplevel)
ARCH_NAME ?= grayskull

include ./module.mk
