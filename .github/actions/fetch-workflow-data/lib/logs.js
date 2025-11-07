// Logs Module
// Handles log processing and workflow matching

/**
 * Check if a workflow name matches any configuration in workflow_configs.
 * @param {string} workflowName - Name of the workflow to check
 * @param {Array} workflowConfigs - Array of config objects with wkflw_name or wkflw_prefix
 * @returns {boolean} True if workflow matches any config
 */
function workflowMatchesConfig(workflowName, workflowConfigs) {
  if (!Array.isArray(workflowConfigs) || workflowConfigs.length === 0) {
    return true; // If no configs provided, match all workflows (backward compatibility)
  }
  for (const config of workflowConfigs) {
    if (config.wkflw_name && workflowName === config.wkflw_name) {
      return true;
    }
    if (config.wkflw_prefix && workflowName.startsWith(config.wkflw_prefix)) {
      return true;
    }
  }
  return false;
}

module.exports = {
  workflowMatchesConfig,
};
