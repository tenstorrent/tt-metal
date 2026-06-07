# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-modality output writers — emit code that writes task output to files.

  * ``text``  -> ``.txt`` transcript / translation
  * ``audio`` -> ``.wav`` via scipy.io.wavfile
  * ``image`` -> ``.png`` via PIL
  * ``scalar`` -> JSON metric file
"""

# These modules emit source-code strings; nothing imported at runtime by demos
