import numpy as np
from models.experimental.BEVFormerV2.projects.mmdet3d_plugin.dependency import PIPELINES
import mmcv


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi-view images from files.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to True.
    """

    def __init__(self, to_float32=True):
        self.to_float32 = to_float32

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after loading, 'img' key is updated.
        """
        filename = results["img_filename"]
        img = []
        print("filename: ", filename)
        for name in filename:
            img_array = mmcv.imread(name, flag="color")
            if self.to_float32:
                img_array = img_array.astype(np.float32)
            img.append(img_array)
        results["img"] = img
        results["img_shape"] = [im.shape for im in img]
        results["ori_shape"] = [im.shape for im in img]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32})"
        return repr_str
