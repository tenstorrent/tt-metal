// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

	// TLS symbols needed in linker script (one cannot set the type there)

	.macro tls name, section
	.pushsection \section,"awT"
	.global \name
	\name:
	.type \name, @object
	.popsection
	.endm

	tls __ldm_tdata_start, .tdata.start
	tls __ldm_tdata_end, .tdata.end
	tls __ldm_tbss_start, .tbss.start
	tls __ldm_tbss_end, .tbss.end
	tls __stack_base_lwm, .tbss.end
