# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The default routed-expert weight dtype, in one place.

`routed_expert_weights_dtype=ttnn.bfloat4_b` used to be spelled as a literal default at each site
that builds or checks the cache. A/B-ing expert precision therefore meant editing every one of them
and hoping none was missed -- and missing one is not a crash, it is a weight cache BUILT at one
dtype and CHECKED at another, which loads the empty placeholder as the weights and produces
plausible output with a meaningless PCC.

One constant, so the build and the completeness check cannot disagree. Callers that want a different
precision pass `routed_expert_weights_dtype` explicitly.

Migrated here: TtMoe, TtPrefillBlock (x2), TtPrefillTransformer, TtPrefillRuntime.

NOT yet migrated: `load_and_compute_layer_by_layer` in utils/transformer_helpers.py still carries a
literal `ttnn.bfloat4_b` default, and none of its four call sites override it. Changing the constant
below therefore leaves that loader building at the old dtype -- the very mismatch this module
exists to prevent. It is left for a follow-up because transformer_helpers.py belongs to a different
PR in this split; migrating it here would put the same file in three PRs at once.
"""

import ttnn

DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE = ttnn.bfloat4_b
