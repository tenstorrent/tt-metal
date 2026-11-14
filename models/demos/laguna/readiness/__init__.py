# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The parts of the model-readiness harness Laguna uses: the Generator contract, the .refpt reference format and
generator, and the prefill / teacher-forcing runners. Copied from the bring-up tool's models/common/readiness_check
(not on main), with imports pointed here, so Laguna does not depend on code outside its folder."""
