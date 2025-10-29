# import torch
# # from torch.autograd import Function
# from ..box_utils import decode, nms
# from models.experimental.SSD512.reference.data import voc as cfg
# import torch.nn as nn

# # class Detect(Function):
# class Detect(nn.Module):
#     """At test time, Detect is the final layer of SSD.  Decode location preds,
#     apply non-maximum suppression to location predictions based on conf
#     scores and threshold to a top_k number of output predictions for both
#     confidence score and locations.
#     """
#     def __init__(self, num_classes, size, bkg_label, top_k, conf_thresh, nms_thresh):
#         self.num_classes = num_classes
#         self.background_label = bkg_label
#         self.top_k = top_k
#         # Parameters used in nms.
#         self.nms_thresh = nms_thresh
#         if nms_thresh <= 0:
#             raise ValueError('nms_threshold must be non negative.')
#         self.conf_thresh = conf_thresh
#         # self.variance = cfg['SSD{}'.format(size)]['variance']
#         # print('variance:',self.variance)

#     # @staticmethod
#     def forward(self, loc_data, conf_data, prior_data):
#         """
#         Args:
#             loc_data: (tensor) Loc preds from loc layers
#                 Shape: [batch,num_priors*4]
#             conf_data: (tensor) Shape: Conf preds from conf layers
#                 Shape: [batch*num_priors,num_classes]
#             prior_data: (tensor) Prior boxes and variances from priorbox layers
#                 Shape: [1,num_priors,4]
#         """
#         num = loc_data.size(0)  # batch size
#         num_priors = prior_data.size(0)
#         output = torch.zeros(num, self.num_classes, self.top_k, 5)
#         print('conf_data size:',conf_data.size())
#         conf_preds = conf_data.transpose(2,1)
#         conf_preds = conf_data.view(num, num_priors,
#                                     self.num_classes).transpose(2, 1)
#         print('conf_preds size:',conf_preds.size())

#         # Decode predictions into bboxes.
#         for i in range(num):
#             decoded_boxes = decode(loc_data[i], prior_data, self.variance)
#             # For each class, perform nms
#             conf_scores = conf_preds[i].clone()

#             for cl in range(1, self.num_classes):
#                 c_mask = conf_scores[cl].gt(self.conf_thresh)
#                 scores = conf_scores[cl][c_mask]
#                 if scores.size(0) == 0:
#                     continue
#                 l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
#                 boxes = decoded_boxes[l_mask].view(-1, 4)
#                 # idx of highest scoring and non-overlapping boxes per class
#                 ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
#                 output[i, cl, :count] = \
#                     torch.cat((scores[ids[:count]].unsqueeze(1),
#                                boxes[ids[:count]]), 1)
#         flt = output.contiguous().view(num, -1, 5)
#         _, idx = flt[:, :, 0].sort(1, descending=True)
#         _, rank = idx.sort(1)
#         flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
#         return output

import torch
import torch.nn as nn
from torchvision.ops import nms  # optional, if torchvision is available


class Detect(nn.Module):
    def __init__(self, num_classes, size, bkg_label, top_k, conf_thresh, nms_thresh):
        super().__init__()  # <-- important
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError("nms_threshold must be non-negative.")
        self.conf_thresh = conf_thresh
        # self.variance = cfg['SSD{}'.format(size)]['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        loc_data: [batch, num_priors, 4] or [batch, num_priors*4]
        conf_data: [batch, num_priors, num_classes] or [batch * num_priors, num_classes]
        prior_data: [1, num_priors, 4] or [num_priors, 4]
        Returns:
            output: [batch, num_classes, top_k, 5] tensor
        """
        device = loc_data.device
        batch = loc_data.size(0)

        # normalize loc_data shape -> [batch, num_priors, 4]
        if loc_data.dim() == 2:
            num_priors = loc_data.size(1) // 4
            loc = loc_data.view(batch, num_priors, 4)
        else:
            loc = loc_data
            num_priors = loc.size(1)

        # Normalize conf_data -> [batch, num_priors, num_classes]
        if conf_data.dim() == 2:
            conf = conf_data.view(batch, num_priors, self.num_classes)
        else:
            conf = conf_data

        # Prior data -> [num_priors, 4]
        if prior_data.dim() == 3 and prior_data.size(0) == 1:
            priors = prior_data.squeeze(0)
        else:
            priors = prior_data

        # Prepare output container
        output = torch.zeros(batch, self.num_classes, self.top_k, 5, device=device)

        # Per-batch, per-class NMS (skip background class 0)
        for i in range(batch):
            for cl in range(1, self.num_classes):
                scores = conf[i, :, cl]  # [num_priors]
                keep_mask = scores > self.conf_thresh
                # if keep_mask.sum().item() == 0:
                #     continue
                if not keep_mask.any():  # still returns a tensor; must avoid .item()
                    continue
                boxes = loc[i][keep_mask]  # [k, 4]  (decode if necessary)
                scores_k = scores[keep_mask]  # [k]

                # If boxes are in center form, decode relative to priors here.
                # Example uses torchvision.nms if boxes are [x1,y1,x2,y2]:
                try:
                    keep_idx = nms(boxes, scores_k, self.nms_thresh)
                except Exception:
                    # fallback: no torchvision or different format - keep top scores
                    _, keep_idx = scores_k.topk(min(self.top_k, scores_k.numel()))
                    keep_idx = keep_idx.cpu()

                # cap top_k
                keep_idx = keep_idx[: self.top_k]

                # for out_i, idx in enumerate(keep_idx):
                #     output[i, cl, out_i, 0] = scores_k[idx]
                #     output[i, cl, out_i, 1:] = boxes[idx]
                K = self.top_k
                pad_idx = torch.zeros(K, dtype=torch.long, device=device)
                n = keep_idx.size(0)
                pad_idx[:n] = keep_idx[:K]
                scores_padded = scores_k[pad_idx]  # [K]
                boxes_padded = boxes[pad_idx]  # [K,4]
                valid_mask = torch.arange(K, device=device) < n
                # place into output in one go:
                output[i, cl, :, 0] = scores_padded * valid_mask.to(scores_padded.dtype)
                output[i, cl, :, 1:] = boxes_padded

        return output
