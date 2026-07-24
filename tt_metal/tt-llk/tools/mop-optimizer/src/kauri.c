// SPDX-FileCopyrightText: © 2026 Zane Hambly
//
// SPDX-License-Identifier: Apache-2.0

/* Copyright (c) 2026 Zane Hambly. Apache License 2.0.
 * See LICENSE for terms. */

/* kauri.c -- the one translation unit that grows the Kauri.
 * Single-header libraries need exactly one home for their guts;
 * this is it. Define it twice and the linker files a complaint. */

#define KAURI_IMPL
#include "kauri.h"
