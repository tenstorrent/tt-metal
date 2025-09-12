"""
Utility functions for working with detected objects.
"""

from loguru import logger


def print_object_comparison(ref_objects, tt_objects):
    """
    Print comparison between reference and TT objects.

    Args:
        ref_objects: List of reference objects
        tt_objects: List of TT objects
    """
    logger.info(f"Reference objects count: {len(ref_objects)}")
    logger.info(f"TTNN objects count: {len(tt_objects)}")

    logger.info("=== Reference Objects ===")
    for i, obj in enumerate(ref_objects):
        logger.info(f"Ref Object {i}: {obj}")

    logger.info("=== TTNN Objects ===")
    for i, obj in enumerate(tt_objects):
        logger.info(f"TT Object {i}: {obj}")

    # Compare object counts and properties if they match
    if len(ref_objects) == len(tt_objects):
        logger.info("=== Object Comparison ===")
        for i, (ref_obj, tt_obj) in enumerate(zip(ref_objects, tt_objects)):
            logger.info(f"Object {i} comparison:")
            logger.info(f"  Classname: {ref_obj.classname} vs {tt_obj.classname}")
            logger.info(f"  Position: {ref_obj.position} vs {tt_obj.position}")
            logger.info(f"  Dimensions: {ref_obj.dimensions} vs {tt_obj.dimensions}")
            logger.info(f"  Angle: {ref_obj.angle} vs {tt_obj.angle}")
            logger.info(f"  Score: {ref_obj.score} vs {tt_obj.score}")
    else:
        logger.warning(f"Object count mismatch: {len(ref_objects)} ref vs {len(tt_objects)} ttnn")
