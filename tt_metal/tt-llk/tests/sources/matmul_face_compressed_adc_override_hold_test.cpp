// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Control for matmul_face_compressed_adc_override_test.cpp: the pack thread holds mutex::THREAD2_ADC across
// each whole pack call, so no pack-thread TDMA-range instruction can coincide with an override SETADC.
#define ADC_PROBE_HOLD
#include "matmul_face_compressed_adc_override_test.cpp"
