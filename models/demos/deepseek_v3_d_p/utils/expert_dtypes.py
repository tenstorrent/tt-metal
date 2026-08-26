# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The default routed-expert weight dtype, in one place.

`routed_expert_weights_dtype=ttnn.bfloat4_b` used to be spelled as a literal default in six places
(TtMoe, TtPrefillBlock x2, TtPrefillTransformer, TtPrefillRuntime, and the layer-by-layer loader).
A/B-ing expert precision therefore meant editing six defaults and hoping none was missed -- and
missing one is not a crash, it is a weight cache BUILT at one dtype and CHECKED at another, which
loads the empty placeholder as the weights and produces plausible output with a meaningless PCC.

One constant, so the build and the completeness check cannot disagree. Callers that want a different
precision pass `routed_expert_weights_dtype` explicitly.
"""

import ttnn

DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE = ttnn.bfloat4_b
