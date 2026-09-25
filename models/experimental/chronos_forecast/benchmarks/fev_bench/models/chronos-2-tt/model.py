# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import math
from pathlib import Path

import datasets
import torch

import fev
import ttnn
from chronos.base import BaseChronosPipeline
from chronos.chronos2.preprocess import PreparedInput, from_data_frame
from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as ReferenceModel
from models.experimental.chronos_forecast.tt.model import TtChronos
from models.experimental.chronos_forecast.tt.program_configs import TtChronosPrecision


class Chronos2TTModel(fev.ForecastingModel):
    """Single-chip TTNN Chronos-2 adapter for fev."""

    model_name = "chronos-2-tt"

    def __init__(
        self,
        model_path: str = "models/experimental/chronos_forecast/weights/chronos-2",
        batch_size: int = 100,
        cross_learning: bool = True,
        as_univariate: bool = False,
        precision: str = "default",
        l1_resident: bool = False,
    ):
        super().__init__()
        if precision not in ("default", "performance"):
            raise ValueError(f"precision must be 'default' or 'performance', got {precision!r}")
        self.model_path = str(Path(model_path).resolve())
        self.batch_size = batch_size
        self.cross_learning = cross_learning
        self.as_univariate = as_univariate
        self._mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
        self._mesh_device.enable_program_cache()
        reference = ReferenceModel.from_pretrained(self.model_path).eval()
        self._quantile_levels = list(reference.chronos_config.quantiles)
        self._output_patch_size = int(reference.chronos_config.output_patch_size)
        tt_precision = TtChronosPrecision.performance() if precision == "performance" else TtChronosPrecision()
        self._model = TtChronos.from_torch_model(
            self._mesh_device,
            reference,
            tt_precision,
            l1_chunk_tokens=tt_precision.l1_chunk_tokens() if l1_resident else None,
        )
        self._closed = False
        atexit.register(self.close)

    def close(self):
        if self._closed:
            return
        ttnn.close_mesh_device(self._mesh_device)
        self._closed = True

    @staticmethod
    def _left_pad_context(items: list[PreparedInput]) -> torch.Tensor:
        max_length = max(item["context"].shape[-1] for item in items)
        rows = []
        for item in items:
            context = item["context"]
            if context.shape[-1] < max_length:
                padding = torch.full(
                    (context.shape[0], max_length - context.shape[-1]),
                    float("nan"),
                    dtype=context.dtype,
                )
                context = torch.cat([padding, context], dim=-1)
            rows.append(context)
        return torch.cat(rows, dim=0)

    def _iter_batches(self, inputs: list[PreparedInput]):
        if not self.cross_learning:
            yield from ([item] for item in inputs)
            return
        batch = []
        rows = 0
        for item in inputs:
            item_rows = int(item["context"].shape[0])
            if batch and rows + item_rows > self.batch_size:
                yield batch
                batch = []
                rows = 0
            batch.append(item)
            rows += item_rows
        if batch:
            yield batch

    def _predict_prepared(self, inputs: list[PreparedInput], horizon: int) -> list[torch.Tensor]:
        predictions: list[torch.Tensor] = []
        num_output_patches = math.ceil(horizon / self._output_patch_size)
        for batch in self._iter_batches(inputs):
            context = self._left_pad_context(batch)
            future_covariates = torch.cat([item["future_covariates"] for item in batch], dim=0)
            group_ids = torch.cat(
                [
                    torch.full((item["context"].shape[0],), group_index, dtype=torch.long)
                    for group_index, item in enumerate(batch)
                ]
            )
            target_ranges = []
            row_offset = 0
            for item in batch:
                target_ranges.append((row_offset, row_offset + item["n_targets"]))
                row_offset += item["context"].shape[0]

            prepared = self._model.prepare_inputs(
                context=context,
                group_ids=group_ids,
                future_covariates=future_covariates,
                num_output_patches=num_output_patches,
            )
            device_inputs = self._model.upload_inputs(prepared)
            output_device = None
            try:
                output_device = self._model.forward_device(device_inputs)
                batch_predictions = self._model.postprocess_output(
                    output_device,
                    prepared.loc_scale,
                    num_output_patches=num_output_patches,
                )
            finally:
                if output_device is not None:
                    ttnn.deallocate(output_device)
                self._model.deallocate_inputs(device_inputs)

            for start, stop in target_ranges:
                predictions.append(batch_predictions[start:stop, :, :horizon].cpu())
        return predictions

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        predictions_per_window = []
        for window in task.iter_windows():
            past_df, future_df, target_columns = BaseChronosPipeline._fev_window_to_df(
                window, as_univariate=self.as_univariate
            )
            prepared_inputs = from_data_frame(
                past_df,
                target_columns=target_columns,
                prediction_length=window.horizon,
                future_df=future_df,
                id_column=window.id_column,
                timestamp_column=window.timestamp_column,
                validate_inputs=False,
            )
            with self._record_inference_time():
                predictions = self._predict_prepared(prepared_inputs, window.horizon)

            quantile_indices = []
            for quantile in task.quantile_levels:
                if quantile not in self._quantile_levels:
                    raise ValueError(
                        f"TTNN fev adapter currently requires trained quantiles; unsupported quantile {quantile}"
                    )
                quantile_indices.append(self._quantile_levels.index(quantile))
            median_index = self._quantile_levels.index(0.5)
            per_target = {}
            for target_index, target_name in enumerate(window.target_columns):
                target_predictions = [prediction[target_index] for prediction in predictions]
                data = {
                    "predictions": [prediction[median_index].numpy() for prediction in target_predictions],
                }
                for quantile, quantile_index in zip(task.quantile_levels, quantile_indices):
                    data[str(quantile)] = [prediction[quantile_index].numpy() for prediction in target_predictions]
                per_target[target_name] = datasets.Dataset.from_dict(data)
            predictions_per_window.append(datasets.DatasetDict(per_target))
        return predictions_per_window
