# SPDX-License-Identifier: Apache-2.0
"""TTNN port of RF-DETR's transformer tail (on device).

Mirrors ``RfDetrForObjectDetection.forward`` after the backbone+projector:
two-stage query selection (group 0, inference) -> 3-layer deformable decoder
(post-LN) -> detection heads.

Input : projector output ``source`` as a channels-last ttnn tensor [1, 1600, 256]
        ( == reference ``source.flatten(2).transpose(1, 2)`` ).
Output: ``logits`` torch [1, 300, 91] and ``pred_boxes`` torch [1, 300, 4].

Config (B=1, single level, INFERENCE => group_detr=1, use only group [0] modules
and the first 300 of query_feat/refpoint_embed):
  d_model=256, num_queries=300, num_classes=91, decoder_layers=3,
  self-attn heads=8 (head_dim 32), cross-attn(deformable) heads=16 (head_dim 16),
  n_levels=1, n_points=2, decoder_ffn_dim=2048, spatial_shape=(40, 40),
  valid_ratios=1 (no padding => invalid_mask all-False => masking skipped),
  num_pos_feats(sine)=d_model//2=128.

Key device idioms / facts (validated on Blackhole chip #1):
  * ``ttnn.topk`` returns (values, UINT16 indices); matches torch top-300 exactly.
  * ``ttnn.embedding(idx, table)`` does row-gather of a [N, C] table (used for the
    two-stage topk coordinate/query gather).
  * ``ttnn.grid_sample`` takes channel-LAST input (N, H, W, C) + grid (N, Hg, Wg, 2)
    and returns (N, Hg, Wg, C); it requires C % 32 == 0, so the 16-wide head_dim is
    zero-padded to 32 and sliced back (bilinear-sampled zeros stay zero).
"""

import math

import torch
import ttnn

H = W = 40
HW = H * W
N_QUERIES = 300
D_MODEL = 256
NUM_CLASSES = 91
N_HEADS_SELF = 8
HEAD_DIM_SELF = D_MODEL // N_HEADS_SELF  # 32
N_HEADS_CROSS = 16
HEAD_DIM_CROSS = D_MODEL // N_HEADS_CROSS  # 16
HEAD_DIM_CROSS_PAD = 32  # grid_sample needs channels % 32 == 0
N_POINTS = 2
NUM_POS_FEATS = D_MODEL // 2  # 128
FFN_DIM = 2048

WEIGHT_DTYPE = ttnn.bfloat16  # bf16 weights preserve box-head precision (bf8 dropped detection IoU); full fp32 unsupported by ttnn topk/embedding
ACT_DTYPE = ttnn.bfloat16


def _lin(linear, device, weight_dtype=WEIGHT_DTYPE):
    """torch nn.Linear -> (ttnn weight [in,out], ttnn bias [1,out] or None)."""
    w = ttnn.from_torch(
        linear.weight.detach().t().contiguous(), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    b = None
    if linear.bias is not None:
        b = ttnn.from_torch(
            linear.bias.detach().reshape(1, -1), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )
    return {"w": w, "b": b}


def _ln(layernorm, device):
    return {
        "w": ttnn.from_torch(layernorm.weight.detach().reshape(1, 1, -1), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device),
        "b": ttnn.from_torch(layernorm.bias.detach().reshape(1, 1, -1), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device),
        "eps": float(layernorm.eps),
    }


def _mlp(mlp, device, weight_dtype=WEIGHT_DTYPE):
    """Reference MLP (relu between layers, no act on last)."""
    return [_lin(layer, device, weight_dtype=weight_dtype) for layer in mlp.layers]


class TtTransformer:
    def __init__(self, ref_model, device):
        self.device = device
        ref = ref_model
        tf = ref.transformer
        cfg = ref.cfg

        # ---- two-stage selection heads (group 0 only) ----
        self.enc_output = _lin(tf.enc_output[0], device)
        self.enc_output_norm = _ln(tf.enc_output_norm[0], device)
        self.enc_out_class_embed = _lin(tf.enc_out_class_embed[0], device)
        self.enc_out_bbox_embed = _mlp(tf.enc_out_bbox_embed[0], device)

        # ---- decoder shared pieces ----
        dec = tf.decoder
        self.ref_point_head = _mlp(dec.ref_point_head, device)
        self.dec_norm = _ln(dec.norm, device)
        self.layers = [self._extract_layer(l, device) for l in dec.layers]

        # ---- final heads ----
        self.class_embed = _lin(ref.class_embed, device)
        self.bbox_embed = _mlp(ref.bbox_embed, device)

        # ---- input-independent constants ----
        # output_proposals [1,1600,4] (invalid_mask is all-False at 40x40 => no masking).
        _, output_proposals, invalid_mask = ref._gen_proposals(
            torch.zeros(1, HW, D_MODEL), torch.zeros(1, HW, dtype=torch.bool), [(H, W)]
        )
        assert not bool(invalid_mask.any()), "invalid_mask must be all-False at 40x40"
        self.output_proposals = ttnn.from_torch(
            output_proposals, dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )

        # refpoint_embed[:300] [300,4], query_feat[:300] [300,256]
        refpoint = ref.refpoint_embed.weight[:N_QUERIES].detach()  # [300,4]
        self.refpoint_embed = ttnn.from_torch(
            refpoint.reshape(1, N_QUERIES, 4), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )
        query_feat = ref.query_feat.weight[:N_QUERIES].detach()  # [300,256]
        self.target = ttnn.from_torch(
            query_feat.reshape(1, N_QUERIES, D_MODEL), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )

        # sine embedding: inv_dim_t [1,1,128] and even/odd selection masks.
        dim_t = torch.arange(NUM_POS_FEATS, dtype=torch.float32)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / NUM_POS_FEATS)
        inv_dim_t = (2 * math.pi) / dim_t  # [128]
        self.inv_dim_t = ttnn.from_torch(
            inv_dim_t.reshape(1, 1, NUM_POS_FEATS), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )
        even = torch.zeros(NUM_POS_FEATS)
        even[0::2] = 1.0
        odd = 1.0 - even
        self.sine_even = ttnn.from_torch(
            even.reshape(1, 1, NUM_POS_FEATS), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.sine_odd = ttnn.from_torch(
            odd.reshape(1, 1, NUM_POS_FEATS), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def _extract_layer(self, layer, device):
        sa = layer.self_attn
        # nn.MultiheadAttention in_proj_weight [768,256] -> q/k/v [256,256] each.
        qw, kw, vw = sa.in_proj_weight.detach().chunk(3, dim=0)
        qb, kb, vb = sa.in_proj_bias.detach().chunk(3, dim=0)

        def _mk(w, b):
            return {
                "w": ttnn.from_torch(w.t().contiguous(), dtype=WEIGHT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device),
                "b": ttnn.from_torch(b.reshape(1, -1), dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device),
            }

        ca = layer.cross_attn
        return {
            "sa_q": _mk(qw, qb),
            "sa_k": _mk(kw, kb),
            "sa_v": _mk(vw, vb),
            "sa_out": _lin(sa.out_proj, device),
            "norm1": _ln(layer.norm1, device),
            "ca_sampling_offsets": _lin(ca.sampling_offsets, device),
            "ca_attention_weights": _lin(ca.attention_weights, device),
            "ca_value_proj": _lin(ca.value_proj, device),
            "ca_output_proj": _lin(ca.output_proj, device),
            "norm2": _ln(layer.norm2, device),
            "linear1": _lin(layer.linear1, device),
            "linear2": _lin(layer.linear2, device),
            "norm3": _ln(layer.norm3, device),
        }

    # ---------------- ops ----------------
    def _layer_norm(self, x, ln):
        return ttnn.layer_norm(x, weight=ln["w"], bias=ln["b"], epsilon=ln["eps"])

    def _linear(self, x, p):
        return ttnn.linear(x, p["w"], bias=p["b"], compute_kernel_config=self.compute_config)

    def _mlp_fwd(self, x, layers):
        for i, p in enumerate(layers):
            x = self._linear(x, p)
            if i < len(layers) - 1:
                x = ttnn.relu(x)
        return x

    def _refine_bboxes(self, reference_points, deltas):
        """cxcy = delta_xy*ref_wh + ref_xy; wh = exp(delta_wh)*ref_wh.  shapes [...,4]."""
        ref_xy = reference_points[..., :2]
        ref_wh = reference_points[..., 2:]
        d_xy = deltas[..., :2]
        d_wh = deltas[..., 2:]
        new_cxcy = ttnn.add(ttnn.multiply(d_xy, ref_wh), ref_xy)
        new_wh = ttnn.multiply(ttnn.exp(d_wh), ref_wh)
        return ttnn.concat([new_cxcy, new_wh], dim=-1)

    def _sine_embed(self, pos):
        """encode_sinusoidal_position_embedding(pos[1,300,4], 128) -> [1,300,512].

        Per coord: full = coord * inv_dim_t [1,300,128];
        out = sin(full)*even_mask + cos(full)*odd_mask. Then swap coords 0/1, concat.
        """
        embs = []
        for c in range(4):
            coord = pos[:, :, c : c + 1]  # [1,300,1]
            full = ttnn.multiply(coord, self.inv_dim_t)  # broadcast -> [1,300,128]
            s = ttnn.sin(full)
            cs = ttnn.cos(full)
            e = ttnn.add(ttnn.multiply(s, self.sine_even), ttnn.multiply(cs, self.sine_odd))
            embs.append(e)
        embs[0], embs[1] = embs[1], embs[0]
        return ttnn.concat(embs, dim=-1)  # [1,300,512]

    def _self_attention(self, hidden, query_pos, p):
        # q = k = hidden + query_pos; v = hidden
        qk_in = ttnn.add(hidden, query_pos)
        q = self._linear(qk_in, p["sa_q"])
        k = self._linear(qk_in, p["sa_k"])
        v = self._linear(hidden, p["sa_v"])
        # reshape to [1, heads, 300, head_dim]
        q = ttnn.transpose(ttnn.reshape(q, (1, N_QUERIES, N_HEADS_SELF, HEAD_DIM_SELF)), 1, 2)
        k = ttnn.transpose(ttnn.reshape(k, (1, N_QUERIES, N_HEADS_SELF, HEAD_DIM_SELF)), 1, 2)
        v = ttnn.transpose(ttnn.reshape(v, (1, N_QUERIES, N_HEADS_SELF, HEAD_DIM_SELF)), 1, 2)
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.compute_config)
        scores = ttnn.multiply(scores, HEAD_DIM_SELF ** -0.5)
        probs = ttnn.softmax(scores, dim=-1)
        ctx = ttnn.matmul(probs, v, compute_kernel_config=self.compute_config)  # [1,heads,300,head_dim]
        ctx = ttnn.reshape(ttnn.transpose(ctx, 1, 2), (1, N_QUERIES, D_MODEL))
        return self._linear(ctx, p["sa_out"])

    def _deformable_attention(self, hidden, query_pos, reference_points, value_proj, p):
        """MSDeformAttn (16 heads, 1 level, 2 points). reference_points [1,300,4]."""
        query = ttnn.add(hidden, query_pos)

        # sampling_offsets -> [1,300,16,1,2,2]; attention_weights -> softmax over 2 points
        offsets = self._linear(query, p["ca_sampling_offsets"])  # [1,300,64]
        offsets = ttnn.reshape(offsets, (1, N_QUERIES, N_HEADS_CROSS, N_POINTS, 2))  # n_levels=1 squeezed
        attn_w = self._linear(query, p["ca_attention_weights"])  # [1,300,32]
        attn_w = ttnn.reshape(attn_w, (1, N_QUERIES, N_HEADS_CROSS, N_POINTS))
        attn_w = ttnn.softmax(attn_w, dim=-1)  # over n_levels*n_points = 2

        # 4-d reference points: loc = ref_xy + offsets / n_points * ref_wh * 0.5
        # ref slices: [1,300,1,1,2] broadcastable to [1,300,16,2,2]
        ref_xy = ttnn.reshape(reference_points[:, :, :2], (1, N_QUERIES, 1, 1, 2))
        ref_wh = ttnn.reshape(reference_points[:, :, 2:], (1, N_QUERIES, 1, 1, 2))
        loc = ttnn.add(ref_xy, ttnn.multiply(ttnn.multiply(offsets, (0.5 / N_POINTS)), ref_wh))  # [1,300,16,2,2]

        # grid_sample core (single level).
        # value [1,1600,256] -> [1,1600,16,16] -> per-head NHWC [16,40,40,16] (pad C->32)
        value = ttnn.reshape(value_proj, (1, HW, N_HEADS_CROSS, HEAD_DIM_CROSS))
        value = ttnn.permute(value, (0, 2, 1, 3))  # [1,16,1600,16]
        value = ttnn.reshape(value, (N_HEADS_CROSS, H, W, HEAD_DIM_CROSS))  # [16,40,40,16]
        value = ttnn.pad(value, [(0, 0), (0, 0), (0, 0), (0, HEAD_DIM_CROSS_PAD - HEAD_DIM_CROSS)], value=0.0)

        # sampling_grids = 2*loc - 1 ; grid [16, 300, 2(points), 2(xy)]
        grids = ttnn.subtract(ttnn.multiply(loc, 2.0), 1.0)  # [1,300,16,2,2]
        grids = ttnn.permute(grids, (0, 2, 1, 3, 4))  # [1,16,300,2,2]
        grids = ttnn.reshape(grids, (N_HEADS_CROSS, N_QUERIES, N_POINTS, 2))  # [16,300,2,2]

        # grid_sample requires ROW_MAJOR layout for both value and grid.
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
        grids = ttnn.to_layout(grids, ttnn.ROW_MAJOR_LAYOUT)
        sampled = ttnn.grid_sample(
            value, grids, mode="bilinear", padding_mode="zeros", align_corners=False
        )  # [16,300,2,32]
        sampled = ttnn.to_layout(sampled, ttnn.TILE_LAYOUT)
        sampled = sampled[:, :, :, :HEAD_DIM_CROSS]  # [16,300,2,16]

        # weight by attention_weights and sum over points.
        # attn_w [1,300,16,2] -> [16,300,2,1]
        aw = ttnn.permute(attn_w, (0, 2, 1, 3))  # [1,16,300,2]
        aw = ttnn.reshape(aw, (N_HEADS_CROSS, N_QUERIES, N_POINTS, 1))
        weighted = ttnn.multiply(sampled, aw)  # [16,300,2,16]
        out = ttnn.sum(weighted, dim=2)  # [16,300,16]
        # -> [1,300,256]: out is per-head [16,300,16]; reshape to [1,16,300,16]->[1,300,16,16]->[1,300,256]
        out = ttnn.reshape(out, (1, N_HEADS_CROSS, N_QUERIES, HEAD_DIM_CROSS))
        out = ttnn.permute(out, (0, 2, 1, 3))  # [1,300,16,16]
        out = ttnn.reshape(out, (1, N_QUERIES, D_MODEL))
        return self._linear(out, p["ca_output_proj"])

    def _decoder_layer(self, hidden, query_pos, reference_points, value_proj, p):
        # self-attention (post-LN)
        attn = self._self_attention(hidden, query_pos, p)
        hidden = self._layer_norm(ttnn.add(hidden, attn), p["norm1"])
        # deformable cross-attention
        cross = self._deformable_attention(hidden, query_pos, reference_points, value_proj, p)
        hidden = self._layer_norm(ttnn.add(hidden, cross), p["norm2"])
        # FFN
        ffn = self._linear(ttnn.relu(self._linear(hidden, p["linear1"])), p["linear2"])
        hidden = self._layer_norm(ttnn.add(hidden, ffn), p["norm3"])
        return hidden

    def __call__(self, source):
        """source: ttnn channels-last [1,1600,256]. Returns (logits, pred_boxes) as torch."""
        device = self.device
        if not isinstance(source, ttnn.Tensor):
            source = ttnn.from_torch(source, dtype=ACT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
        source = ttnn.to_layout(source, ttnn.TILE_LAYOUT)
        if source.dtype != ACT_DTYPE:
            source = ttnn.typecast(source, ACT_DTYPE)

        # ---- two-stage proposal heads ----
        object_query = self._layer_norm(self._linear(source, self.enc_output), self.enc_output_norm)
        enc_class = self._linear(object_query, self.enc_out_class_embed)  # [1,1600,91]
        delta_bbox = self._mlp_fwd(object_query, self.enc_out_bbox_embed)  # [1,1600,4]
        enc_coord = self._refine_bboxes(self.output_proposals, delta_bbox)  # [1,1600,4]

        # scores = enc_class.max(-1) -> [1,1600]; topk 300
        scores = ttnn.max(enc_class, dim=-1)  # [1,1600]
        scores = ttnn.reshape(scores, (1, HW))
        if scores.dtype != ttnn.bfloat16:  # topk requires bf16/bf8 input (ranking only)
            scores = ttnn.typecast(scores, ttnn.bfloat16)
        _, topk_idx = ttnn.topk(scores, N_QUERIES, dim=-1)  # idx [1,300] uint16
        topk_idx = ttnn.typecast(topk_idx, ttnn.uint32)

        # gather rows of enc_coord [1,1600,4] by topk_idx -> [1,300,4] (embedding row-gather)
        enc_coord_tbl = ttnn.to_layout(ttnn.reshape(enc_coord, (HW, 4)), ttnn.TILE_LAYOUT)
        idx_rm = ttnn.to_layout(topk_idx, ttnn.ROW_MAJOR_LAYOUT)
        topk_coords = ttnn.embedding(idx_rm, enc_coord_tbl)  # [1,300,4]
        topk_coords = ttnn.to_layout(topk_coords, ttnn.TILE_LAYOUT)

        # reference_points = refine_bboxes(topk_coords, refpoint_embed[:300]) -> [1,300,4]
        reference_points = self._refine_bboxes(topk_coords, self.refpoint_embed)
        init_reference_points = reference_points

        # ---- decoder query positional embedding (lite refine: ref fixed across layers) ----
        # valid_ratios=1 => ref_inputs == reference_points (single level).
        query_sine = self._sine_embed(reference_points)  # [1,300,512]
        query_pos = self._mlp_fwd(query_sine, self.ref_point_head)  # [1,300,256]

        # cross-attn value projection (same for every layer).
        value_proj = self._linear(source, self.layers[0]["ca_value_proj"])

        hidden = self.target
        last_intermediate = None
        for li, p in enumerate(self.layers):
            if li > 0:
                value_proj = self._linear(source, p["ca_value_proj"])
            hidden = self._decoder_layer(hidden, query_pos, reference_points, value_proj, p)
            last_intermediate = self._layer_norm(hidden, self.dec_norm)

        # ---- heads ----
        logits = self._linear(last_intermediate, self.class_embed)  # [1,300,91]
        boxes_delta = self._mlp_fwd(last_intermediate, self.bbox_embed)  # [1,300,4]
        pred_boxes = self._refine_bboxes(init_reference_points, boxes_delta)  # [1,300,4]

        logits_t = ttnn.to_torch(logits).float().reshape(1, N_QUERIES, NUM_CLASSES)
        pred_boxes_t = ttnn.to_torch(pred_boxes).float().reshape(1, N_QUERIES, 4)
        return logits_t, pred_boxes_t
