# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-modality input synthesizers — reused by task templates.

Each module emits Python SOURCE CODE that becomes part of the
generated demo. The synthesizer modules are NOT imported at runtime
by the emitted demo; they're string-building utilities used at
emit time.

  * ``audio`` — librosa load + VAD silence split + fbank features
  * ``image`` — PIL load + torchvision transforms + normalize
  * ``text`` — tokenizer + prompt template
  * ``video`` — cv2/decord frame extract (future)
  * ``csv_timeseries`` — pandas + gluonts time features (future)
"""
from . import audio, image, text  # noqa: F401
