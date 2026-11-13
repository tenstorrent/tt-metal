/* SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* Added to firmware_drisc.ld at link time. drisc_service.cpp defines drisc_service_text_limit. */
ASSERT(__fw_export_text_end <= drisc_service_text_limit, "drisc_service: .text overlaps the batch windows")
