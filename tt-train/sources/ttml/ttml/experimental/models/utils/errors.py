# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0


class UnsupportedTrainingConfig(ValueError):
    """This model can't be trained with this config. The message names the setting and says why."""
