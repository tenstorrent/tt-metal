# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import os, json
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


def loader_func(path):
    return Image.open(path)


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1.0, 0.0)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception("Format of lanes error.")
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0.0, 0.0, 1.0
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0.0, 0.0
        matched = 0.0
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.0
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return (
            s / max(min(4.0, len(gt)), 1.0),
            fp / len(pred) if len(pred) > 0 else 0.0,
            fn / max(min(len(gt), 4.0), 1.0),
        )

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        json_pred = [json.loads(line) for line in open(pred_file, "r").readlines()]
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception("We do not get the predictions of all the test tasks")
        gts = {l["raw_file"]: l for l in json_gt}
        accuracy, fp, fn = 0.0, 0.0, 0.0
        for pred in json_pred:
            if "raw_file" not in pred or "lanes" not in pred or "run_time" not in pred:
                raise Exception("raw_file or lanes or run_time not in some predictions.")
            raw_file = pred["raw_file"]
            pred_lanes = pred["lanes"]
            run_time = pred["run_time"]
            if raw_file not in gts:
                raise Exception("Some raw_file from your predictions do not exist in the test tasks.")
            gt = gts[raw_file]
            gt_lanes = gt["lanes"]
            y_samples = gt["h_samples"]
            try:
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception("Format of lanes error.")
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        pr = 1 - fp / num
        re = 1 - fn / num
        if (pr + re) == 0:
            f1 = 0
        else:
            f1 = 2 * pr * re / (pr + re)
        return json.dumps(
            [
                {"name": "Accuracy", "value": accuracy / num, "order": "desc"},
                {"name": "FP", "value": fp / num, "order": "asc"},
                {"name": "FN", "value": fn / num, "order": "asc"},
                {"name": "F1", "value": f1, "order": "asc"},
            ]
        )


class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None, crop_size=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        self.crop_size = crop_size
        with open(list_path, "r") as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == "/" else l for l in self.list]

    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)
        img = img[:, -self.crop_size :, :]

        return img, name

    def __len__(self):
        return len(self.list)


def get_test_loader(batch_size, data_root, dataset, distributed, crop_ratio, train_width, train_height):
    if dataset == "Tusimple":
        img_transforms = transforms.Compose(
            [
                transforms.Resize((int(train_height / crop_ratio), train_width)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_dataset = LaneTestDataset(
            os.path.join(data_root, "images"),
            os.path.join(data_root, "input_images.txt"),
            img_transform=img_transforms,
            crop_size=train_height,
        )
    else:
        raise NotImplementedError

    sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return loader


def generate_tusimple_lines(row_out, row_ext, col_out, col_ext, row_anchor=None, col_anchor=None, mode="2row2col"):
    tusimple_h_sample = np.linspace(160, 710, 56)
    row_num_grid, row_num_cls, row_num_lane = row_out.shape
    row_max_indices = row_out.argmax(0).cpu()
    row_valid = row_ext.argmax(0).cpu()
    row_out = row_out.cpu()

    col_num_grid, col_num_cls, col_num_lane = col_out.shape
    col_max_indices = col_out.argmax(0).cpu()
    col_valid = col_ext.argmax(0).cpu()
    col_out = col_out.cpu()

    if mode == "normal" or mode == "2row2col":
        row_lane_list = [1, 2]
        col_lane_list = [0, 3]
    elif mode == "4row":
        row_lane_list = range(row_num_lane)
        col_lane_list = []
    elif mode == "4col":
        row_lane_list = []
        col_lane_list = range(col_num_lane)
    else:
        raise NotImplementedError

    local_width_row = 14
    local_width_col = 14
    min_lanepts_row = 3
    min_lanepts_col = 3

    all_lanes = []

    for row_lane_idx in row_lane_list:
        if row_valid[:, row_lane_idx].sum() > min_lanepts_row:
            cur_lane = []
            for row_cls_idx in range(row_num_cls):
                if row_valid[row_cls_idx, row_lane_idx]:
                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, row_max_indices[row_cls_idx, row_lane_idx] - local_width_row),
                                min(row_num_grid - 1, row_max_indices[row_cls_idx, row_lane_idx] + local_width_row) + 1,
                            )
                        )
                    )
                    coord = (row_out[all_ind, row_cls_idx, row_lane_idx].softmax(0) * all_ind.float()).sum() + 0.5
                    coord_x = coord / (row_num_grid - 1) * 1280
                    coord_y = row_anchor[row_cls_idx] * 720
                    cur_lane.append(int(coord_x))
                else:
                    cur_lane.append(-2)
            all_lanes.append(cur_lane)
        else:
            pass

    for col_lane_idx in col_lane_list:
        if col_valid[:, col_lane_idx].sum() > min_lanepts_col:
            cur_lane = []
            for col_cls_idx in range(col_num_cls):
                if col_valid[col_cls_idx, col_lane_idx]:
                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, col_max_indices[col_cls_idx, col_lane_idx] - local_width_col),
                                min(col_num_grid - 1, col_max_indices[col_cls_idx, col_lane_idx] + local_width_col) + 1,
                            )
                        )
                    )
                    coord = (col_out[all_ind, col_cls_idx, col_lane_idx].softmax(0) * all_ind.float()).sum() + 0.5
                    coord_y = coord / (col_num_grid - 1) * 720
                    coord_x = col_anchor[col_cls_idx] * 1280
                    cur_lane.append((coord_x, coord_y))
            cur_lane = np.array(cur_lane)
            top_lim = min(cur_lane[:, 1])
            bot_lim = max(cur_lane[:, 1])

            p = np.polyfit(cur_lane[:, 1], cur_lane[:, 0], deg=2)
            lanes_on_tusimple = np.polyval(p, tusimple_h_sample)

            lanes_on_tusimple = np.round(lanes_on_tusimple)
            lanes_on_tusimple = lanes_on_tusimple.astype(int)
            lanes_on_tusimple[lanes_on_tusimple < 0] = -2
            lanes_on_tusimple[lanes_on_tusimple > 1280] = -2
            lanes_on_tusimple[tusimple_h_sample < top_lim] = -2
            lanes_on_tusimple[tusimple_h_sample > bot_lim] = -2
            all_lanes.append(lanes_on_tusimple.tolist())
        else:
            pass
    return all_lanes


def run_test_tusimple(
    net,
    data_root,
    work_dir,
    exp_name,
    distributed,
    crop_ratio,
    train_width,
    train_height,
    batch_size=1,
    row_anchor=None,
    col_anchor=None,
    device=None,
    num_cls_row=56,
    num_grid_col=100,
    num_cls_col=41,
    num_lane_on_row=4,
    num_lane_on_col=4,
    num_grid_row=100,
):
    output_path = os.path.join(work_dir, exp_name + ".txt")
    fp = open(output_path, "w")
    loader = get_test_loader(batch_size, data_root, "Tusimple", distributed, crop_ratio, train_width, train_height)

    dim1 = num_grid_row * num_cls_row * num_lane_on_row
    dim2 = num_grid_col * num_cls_col * num_lane_on_col
    dim3 = 2 * num_cls_row * num_lane_on_row
    dim4 = 2 * num_cls_col * num_lane_on_col
    for data in tqdm(loader, desc="Processing images", ncols=100):
        imgs, names = data
        if exp_name == "reference_model_results":
            with torch.no_grad():
                out, pred = net(imgs)
        else:
            imgs = imgs.permute(0, 2, 3, 1)
            imgs = ttnn.from_torch(imgs, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            out = net(imgs, batch_size=batch_size)
            out = ttnn.to_torch(out).squeeze(dim=0).squeeze(dim=0)
            pred = {
                "loc_row": out[:, :dim1].view(-1, num_grid_row, num_cls_row, num_lane_on_row),
                "loc_col": out[:, dim1 : dim1 + dim2].view(-1, num_grid_col, num_cls_col, num_lane_on_col),
                "exist_row": out[:, dim1 + dim2 : dim1 + dim2 + dim3].view(-1, 2, num_cls_row, num_lane_on_row),
                "exist_col": out[:, -dim4:].view(-1, 2, num_cls_col, num_lane_on_col),
            }

        for b_idx, name in enumerate(names):
            tmp_dict = {}
            tmp_dict["lanes"] = generate_tusimple_lines(
                pred["loc_row"][b_idx],
                pred["exist_row"][b_idx],
                pred["loc_col"][b_idx],
                pred["exist_col"][b_idx],
                row_anchor=row_anchor,
                col_anchor=col_anchor,
                mode="4row",
            )
            tmp_dict["h_samples"] = [
                160,
                170,
                180,
                190,
                200,
                210,
                220,
                230,
                240,
                250,
                260,
                270,
                280,
                290,
                300,
                310,
                320,
                330,
                340,
                350,
                360,
                370,
                380,
                390,
                400,
                410,
                420,
                430,
                440,
                450,
                460,
                470,
                480,
                490,
                500,
                510,
                520,
                530,
                540,
                550,
                560,
                570,
                580,
                590,
                600,
                610,
                620,
                630,
                640,
                650,
                660,
                670,
                680,
                690,
                700,
                710,
            ]
            tmp_dict["raw_file"] = name
            tmp_dict["run_time"] = 10
            json_str = json.dumps(tmp_dict)

            fp.write(json_str + "\n")
    fp.close()
