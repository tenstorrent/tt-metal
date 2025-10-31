from typing import Tuple

from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes import NuScenes
from nuscenes.eval.detection.data_classes import (
    DetectionConfig,
    DetectionMetrics,
    DetectionBox,
    DetectionMetricDataList,
)

import numpy as np


class NuScenesEval_custom(NuScenesEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """

    def __init__(
        self,
        nusc: NuScenes,
        config: DetectionConfig,
        result_path: str,
        eval_set: str,
        output_dir: str = None,
        verbose: bool = True,
        overlap_test=False,
        eval_mask=False,
        data_infos=None,
    ):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """

        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.overlap_test = overlap_test
        self.eval_mask = eval_mask
        self.data_infos = data_infos
        # Check result file exists.
        assert os.path.exists(result_path), "Error: The result file does not exist!"

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, "plots")
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print("Initializing nuScenes detection evaluation")
        self.pred_boxes, self.meta = load_prediction(
            self.result_path, self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose
        )
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox_modified, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(
            self.gt_boxes.sample_tokens
        ), "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).

        if verbose:
            print("Filtering predictions")
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print("Filtering ground truth annotations")
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        if self.overlap_test:
            self.pred_boxes = filter_eval_boxes_by_overlap(self.nusc, self.pred_boxes)

            self.gt_boxes = filter_eval_boxes_by_overlap(self.nusc, self.gt_boxes, verbose=True)

        self.all_gt = copy.deepcopy(self.gt_boxes)
        self.all_preds = copy.deepcopy(self.pred_boxes)
        self.sample_tokens = self.gt_boxes.sample_tokens

        self.index_map = {}
        for scene in nusc.scene:
            first_sample_token = scene["first_sample_token"]
            sample = nusc.get("sample", first_sample_token)
            self.index_map[first_sample_token] = 1
            index = 2
            while sample["next"] != "":
                sample = nusc.get("sample", sample["next"])
                self.index_map[sample["token"]] = index
                index += 1

    def update_gt(self, type_="vis", visibility="1", index=1):
        if type_ == "vis":
            self.visibility_test = True
            if self.visibility_test:
                """[{'description': 'visibility of whole object is between 0 and 40%',
                'token': '1',
                'level': 'v0-40'},
                {'description': 'visibility of whole object is between 40 and 60%',
                'token': '2',
                'level': 'v40-60'},
                {'description': 'visibility of whole object is between 60 and 80%',
                'token': '3',
                'level': 'v60-80'},
                {'description': 'visibility of whole object is between 80 and 100%',
                'token': '4',
                'level': 'v80-100'}]"""

                self.gt_boxes = filter_eval_boxes_by_visibility(self.all_gt, visibility, verbose=True)

        elif type_ == "ord":
            valid_tokens = [key for (key, value) in self.index_map.items() if value == index]
            # from IPython import embed
            # embed()
            self.gt_boxes = filter_by_sample_token(self.all_gt, valid_tokens)
            self.pred_boxes = filter_by_sample_token(self.all_preds, valid_tokens)
        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print("Accumulating metric data...")
        metric_data_list = DetectionMetricDataList()

        # print(self.cfg.dist_fcn_callable, self.cfg.dist_ths)
        # self.cfg.dist_ths = [0.3]
        # self.cfg.dist_fcn_callable
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print("Calculating metrics...")
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)
            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ["traffic_cone"] and metric_name in ["attr_err", "vel_err", "orient_err"]:
                    tp = np.nan
                elif class_name in ["barrier"] and metric_name in ["attr_err", "vel_err"]:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print("Rendering PR and TP curves")

        def savepath(name):
            return os.path.join(self.plot_dir, name + ".pdf")

        summary_plot(
            md_list,
            metrics,
            min_precision=self.cfg.min_precision,
            min_recall=self.cfg.min_recall,
            dist_th_tp=self.cfg.dist_th_tp,
            savepath=savepath("summary"),
        )

        for detection_name in self.cfg.class_names:
            class_pr_curve(
                md_list,
                metrics,
                detection_name,
                self.cfg.min_precision,
                self.cfg.min_recall,
                savepath=savepath(detection_name + "_pr"),
            )

            class_tp_curve(
                md_list,
                metrics,
                detection_name,
                self.cfg.min_recall,
                self.cfg.dist_th_tp,
                savepath=savepath(detection_name + "_tp"),
            )

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(
                md_list,
                metrics,
                dist_th,
                self.cfg.min_precision,
                self.cfg.min_recall,
                savepath=savepath("dist_pr_" + str(dist_th)),
            )
