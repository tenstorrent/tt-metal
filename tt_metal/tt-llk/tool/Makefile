# SPDX-FileCopyrightText: © 2026 Zane Hambly
#
# SPDX-License-Identifier: Apache-2.0

# tt-mop -- the open Tensix MOP/replay optimiser
#
# Build the tool:        make
# Run the test suite:    make test
#
# Second piece of the toolchain (ttas assembles, tt-mop optimises,
# tt-emu emulates). What builds today is the expander and verifier:
# the trustworthy core every later search pass is scored against.

CC = gcc

WARN  = -Wall -Wextra -Werror -pedantic
WARN += -Wshadow -Wconversion -Wdouble-promotion -Wundef
WARN += -Wformat=2 -Wnull-dereference -Wswitch-enum -Wswitch-default
WARN += -Wstrict-prototypes -Wold-style-definition -Wmissing-prototypes
WARN += -Wredundant-decls -Wnested-externs -Wcast-align
WARN += -Wno-unused-parameter

INCS   = -Iinclude
CFLAGS = $(WARN) -std=c99 -O2 $(INCS)
TFLAGS = $(WARN) -std=c99 -O0 -g $(INCS)

ifdef DEBUG
CFLAGS = $(WARN) -std=c99 -O0 -g -DDEBUG $(INCS)
TFLAGS = $(WARN) -std=c99 -O0 -g -DDEBUG $(INCS)
endif

TARGET      = ttmop
TEST_TARGET = trunner

ifeq ($(OS),Windows_NT)
TARGET      := $(TARGET).exe
TEST_TARGET := $(TEST_TARGET).exe
endif

SRCS = src/main.c \
       src/kauri.c \
       src/ttmop_expand.c \
       src/ttmop_opt.c
OBJS = $(SRCS:.c=.o)

# Rebuild every object when a header changes: the optimiser and expander
# bake in compile-time constants (assumptions, cost model) from ttmop.h,
# so a stale .o silently runs the old behaviour.
$(OBJS): include/ttmop.h include/kauri.h

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Tests reuse every object except main.o (it owns its own main()).
# ttsim_ref.c is the independent oracle the fuzzer differential-tests against.
TEST_SRCS = tests/tmain.c tests/texpand.c tests/topt.c tests/tmopfit.c \
            tests/tanchor.c tests/tfuzz.c tests/ttsim_ref.c
TEST_OBJS = $(filter-out src/main.o,$(OBJS))

$(TEST_TARGET): $(TEST_SRCS) $(TEST_OBJS)
	$(CC) $(TFLAGS) -o $@ $^ -lm

test: $(TARGET) $(TEST_TARGET)
	./$(TEST_TARGET) --all

clean:
	rm -f $(OBJS) $(TARGET) $(TEST_TARGET) tests/*.o
